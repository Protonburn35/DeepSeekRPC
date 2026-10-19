#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DeepSeekRPC
setup_root=/workspace/deepseekrpc-tools
mkdir -p "$setup_root"
export UV_CACHE_DIR="$setup_root/uv-cache" PIP_CACHE_DIR="$setup_root/pip-cache"
export UV_PYTHON_DOWNLOADS=never UV_NO_PROGRESS=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
command -v python3.12 >/dev/null
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
cp requirements-dev.lock "$setup_root/requirements-dev.lock"
cp requirements-reference-linux.lock "$setup_root/requirements-reference-linux.lock"
if [ ! -x "$setup_root/.venv/bin/python" ]; then
  python3.12 -m uv --system-certs --no-config venv --python python3.12 "$setup_root/.venv"
fi
# uv installs small tools; pip handles the hash-pinned CPU Torch wheel directly.
python3.12 -m uv --system-certs --no-config pip install --python "$setup_root/.venv/bin/python" --default-index https://pypi.org/simple --only-binary :all: --require-hashes -r "$setup_root/requirements-dev.lock"
"$setup_root/.venv/bin/python" -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r "$setup_root/requirements-reference-linux.lock"
source "$setup_root/.venv/bin/activate"
# Existing implementation checkouts get current metadata without resolving a GPU stack.
if [ -f pyproject.toml ]; then
  python -m pip install --no-deps --no-build-isolation -e .
fi
python -m pip check
python - <<'CHECK'
import sys, numpy as np, torch, safetensors, tokenizers, pytest
assert sys.version_info[:2] == (3,12)
assert np.__version__ == '2.3.5'
assert torch.__version__ == '2.13.0+cpu' and torch.version.cuda is None
np.testing.assert_allclose(np.linalg.solve(np.eye(8), np.arange(8)), np.arange(8))
print('PASS: pinned CPU tools and mathematical reference stack; no CUDA qualification')
CHECK
if [ -d tests ] && [ -d src/deepseekrpc ]; then
  python -m pytest -q
fi
cmake --version
ninja --version
native_check=$(mktemp -d "$setup_root/native-check.XXXXXX")
trap 'rm -rf -- "$native_check"' EXIT
cat > "$native_check/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(deepseekrpc_environment_check LANGUAGES CXX)
add_executable(environment_check main.cpp)
target_compile_features(environment_check PRIVATE cxx_std_20)
enable_testing()
add_test(NAME native_toolchain COMMAND environment_check)
CMAKE
cat > "$native_check/main.cpp" <<'CPP'
#include <array>
#include <iostream>
#include <numeric>
int main() {
    constexpr std::array<int, 4> values{2, 3, 5, 7};
    if (std::accumulate(values.begin(), values.end(), 0) != 17) return 1;
    std::cout << "PASS: C++20 compile, link, and execution\n";
    return 0;
}
CPP
cmake -S "$native_check" -B "$native_check/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$(command -v g++)"
cmake --build "$native_check/build" --parallel 2
ctest --test-dir "$native_check/build" --output-on-failure
printf '%s\n' 'DeepSeekRPC cloud CPU development setup complete; no services started.'
