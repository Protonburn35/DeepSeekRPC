# Isolated Windows installation

All three PCs require Windows 11 x64 and a driver supporting the pinned Torch
CUDA 13.0 runtime. A/B use SM89 and C uses SM120. This application does not need
a locally installed CUDA compiler SDK. GPU/model qualification has not yet run.
Use this procedure after the complete source has passed the planned reviews.

Extract the complete `windows-runtime-<commit>` artifact into a new dedicated
directory on each PC. Do not unpack over another version or the Qwen setup.
Keep `BUNDLE.json` and every listed file together. From Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install.ps1
```

The installer verifies bundle files, obtains checksum-pinned uv and CPython
3.12.14, creates the private `runtime` environment and installs hash-locked
dependencies plus the five actual application/native/plugin wheels. It copies
the admitted MSVC CRT beside the private interpreter. It changes neither the
user/machine PATH nor Python registration. Its completion marker is published
only after dependency, native DLL, source identity, concrete class and original
parser checks pass with a bare Windows system PATH. It also builds and loads
a real Python extension with bundled TinyCC and compiles explicit SM89/SM120
Triton inputs to SASS without initializing a GPU context. The pinned standalone
Python includes its development headers/import library; Triton's verified wheel
provides its C compiler, ptxas and minimal CUDA inputs. Inherited compiler,
interpreter and kernel override settings are removed or refused. These checks
hide GPUs and establish installed imports and compiler operation, not model or
GPU correctness. Launchers
refuse an incomplete or mismatched installation. The installer requires network
access to the pinned Python/PyPI/PyTorch artifacts; it downloads no model data.

Generate one private deployment credential set on A:

```powershell
.\runtime\Scripts\python.exe -I -m deepseekrpc.cli init-credentials --output .\credentials
```

The output contains a fresh opaque cluster ID, its private CA, role-specific
mTLS certificates and A's API key. Keep the CA signing key and API key on A.
Securely copy only each worker's role directory to that worker; the role
directory contains its CA certificate, own certificate and own private key.
Never commit, attach or put these private files in diagnostics. Windows files
receive a private current-user/SYSTEM ACL. Generation refuses an existing
output directory. This does not create firewall rules automatically.

Copy `config/cluster.example.json` to `config/cluster.json`, enter the generated
cluster ID and actual worker IPv4 addresses, then copy the same shared JSON to
all three PCs. C's example address comes from the supplied deployment notes;
verify it locally. Copy each role example to `A.json`, `B.json` or `C.json` and
enter local model, state and credential paths. All roles must agree on the
shared file. Keep model files read-only and runtime state in separate directories.
Use distinct state roots for roles and versions. A requires the original Engram
directory; B/C do not. Every unresolved placeholder is refused.

The exact model and original Engram revisions, file names, lengths and publisher
hashes are packaged in the application's deployment manifest. Acquire their
payloads locally on the intended PCs from those publishers. Do not rename,
repack or substitute a different October checkpoint. Startup verifies complete
selected shard hashes using bounded reads, loads only owned tensors, and closes
B's source after its all-page locked preload. Boot verification may take time.
No full model or Engram payload is included in this artifact or cloud checkout.

Run metadata checks, then small local GPU diagnostics before loading:

```powershell
.\launchers\RUN-HOST.bat --config .\config\A.json --preflight
.\launchers\RUN-HOST.bat --config .\config\A.json --self-test
```

Use the corresponding B/C launcher and role configuration on those PCs. The
self-test uses bounded synthetic data and actual native operations against CPU
equations, not a complete-model correctness oracle. Normal startup repeats
required local checks before admitting loaded model memory. Start C, then B,
then A with `--config` and no diagnostic flag. Each process loads once and
retains its weights. HTTP diagnostic readiness exists while A loads. READY
requires all three real stages and matching component/precision/plan identity.

DeepSeek HTTP defaults to `127.0.0.1:8081`; a private-LAN bind requires the local
API key. An occupied port is an error. No existing Qwen listener is stopped or
modified. In Docker, the separate DeepSeek client provider uses
`http://host.docker.internal:8081/v1`; allow only the required private access.
Advertised context is 1,048,576; the default output limit is 8,192 and consumes
that same context budget. Output limits and context capacity are separate.
Unsupported request controls fail clearly rather than being ignored.

Ctrl+C cancels the current request and keeps loaded weights. Ctrl+Break requests
process shutdown. Stop A admission first, then stop workers after requests
quiesce. Interrupted network operations invalidate the session and require a
fresh authenticated mesh before another request; they do not reload weights.
See SESSIONS.md for pause, snapshots and exact pending-input continuation.

To roll back, stop the processes and start the previous separately extracted
bundle with its own state/configuration. Do not copy a completion marker or
runtime directory between versions. The Qwen profile remains independently
available. Check SOURCE.md, THIRD_PARTY_NOTICES.md and component licenses when
redistributing the application or exposing its network service.
