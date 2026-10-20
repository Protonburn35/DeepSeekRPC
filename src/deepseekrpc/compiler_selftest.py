"""Actual private Windows host/device compiler checks without a GPU context."""
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import platform
import re
import struct
import sys

from .jsonutil import ValidationError
from .triton_policy import assert_windows_triton, configure_windows_triton

_C_SOURCE = r'''
#include <Python.h>
static PyObject *answer(PyObject *self, PyObject *ignored) {
    return PyLong_FromLong(41017);
}
static PyMethodDef methods[] = {
    {"answer", answer, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "deepseekrpc_compiler_probe", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit_deepseekrpc_compiler_probe(void) {
    return PyModule_Create(&module);
}
'''


def cubin_architecture(binary):
    """Decode the documented CUDA ELF ABI 7/8 architecture field.

    Blackwell ABI 8 moves EF_CUDA_SM from bits 0..7 to bits 8..15.
    See provenance/compiler-elf-reference.lock.json; no LLVM code is copied.
    """
    if (len(binary) < 64 or binary[:6] != b"\x7fELF\x02\x01" or
            struct.unpack_from("<H", binary, 18)[0] != 190 or binary[8] not in (7, 8)):
        raise ValidationError("offline output is not supported little-endian 64-bit CUDA ELF")
    flags = struct.unpack_from("<I", binary, 48)[0]
    return (flags >> 8) & 0xff if binary[8] == 8 else flags & 0xff


def run_compiler_checks(cache_root=None):
    policy = configure_windows_triton(cache_root)
    lock = json.loads(resources.files("deepseekrpc._vendor.vllm_windows").joinpath("TRITON.json").read_text())
    if platform.python_version() != lock["python_version"]:
        raise ValidationError("compiler qualification requires the exact private Python interpreter")
    headers = {}
    for record in lock["python_headers"]:
        path = Path(sys.base_prefix) / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValidationError("pinned standalone Python development input is absent")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValidationError("standalone Python development input differs from its verified archive")
        headers[record["path"]] = digest
    from triton import compile as compile_kernel, knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime import driver
    from triton.runtime.build import compile_module_from_src, get_cc
    from triton.windows_utils import find_cuda
    from ._triton_compile_probe import copy_probe
    if driver._active is not None or driver._default is not None:
        raise ValidationError("offline compiler qualification must not create a GPU driver/context")
    if (Path(get_cc()).resolve() != Path(os.environ["CC"]).resolve() or
            Path(knobs.nvidia.ptxas.path).resolve() != Path(os.environ["TRITON_PTXAS.EXE_PATH"]).resolve()):
        raise ValidationError("actual Triton compiler selection differs from the admitted tools")
    cuda_bin, includes, libraries = find_cuda()
    cuda = Path(os.environ["CUDA_PATH"])
    if (Path(cuda_bin).resolve() != (cuda / "bin").resolve() or
            [Path(p).resolve() for p in includes] != [(cuda / "include").resolve()] or
            [Path(p).resolve() for p in libraries] != [(cuda / "lib/x64").resolve()]):
        raise ValidationError("Triton selected external CUDA headers or import libraries")
    module = compile_module_from_src(_C_SOURCE, "deepseekrpc_compiler_probe")
    if module.answer() != 41017:
        raise AssertionError("actual TinyCC extension did not load/execute with standalone Python headers")
    with Path(module.__file__).open("rb") as stream:
        host_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    compiled = []
    for architecture in (89, 120):
        source = ASTSource(copy_probe, signature={"source": "*fp32", "destination": "*fp32", "count": "constexpr"},
                           constexprs={"count": 32})
        kernel = compile_kernel(source, target=GPUTarget("cuda", architecture, 32), options={"num_warps": 4})
        ptx, binary = kernel.asm["ptx"], kernel.asm["cubin"]
        observed = {"requested": architecture, "ptx_targets": re.findall(r"\.target\s+(\w+)", ptx),
            "header": binary[:16].hex(), "bytes": len(binary),
            "machine": struct.unpack_from("<H", binary, 18)[0] if len(binary) >= 20 else None,
            "flags": struct.unpack_from("<I", binary, 48)[0] if len(binary) >= 52 else None}
        if (not re.search(rf"\.target\s+sm_{architecture}(?:[af])?\b", ptx) or
                cubin_architecture(binary) != architecture):
            raise AssertionError("offline Triton output failed architecture inspection: " + json.dumps(observed, sort_keys=True))
        compiled.append({"architecture": f"sm_{architecture}", "cubin_bytes": len(binary),
            "elf_abi": binary[8], "elf_flags": observed["flags"],
            "cubin_sha256": hashlib.sha256(binary).hexdigest(), "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "status": "compiled; not GPU-executed"})
    # Exercise the real initialized-policy guard, then restore it exactly.
    for name, value in (("TRITON_INTERPRET", "1"), ("TRITON_KERNEL_OVERRIDE", "1"), ("TRITON_OVERRIDE_ARCH", "sm_80")):
        original = os.environ.get(name)
        try:
            os.environ[name] = value
            try:
                assert_windows_triton()
            except ValidationError:
                pass
            else:
                raise AssertionError("a changed Triton interpreter/kernel/architecture policy was admitted")
        finally:
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
    assert_windows_triton()
    if driver._active is not None or driver._default is not None:
        raise AssertionError("offline compilation unexpectedly initialized a GPU driver")
    return {"scope": "actual Windows TinyCC extension build/load and explicit-target offline Triton compilation",
        "python_headers": headers, "compiler_policy": policy, "host_extension_sha256": host_sha256,
        "targets": compiled, "mutation_guards": "passed", "GPU_execution": "not run"}
