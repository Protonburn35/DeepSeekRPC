# Separate OpenCode provider

`config/opencode.deepseek.example.json` is a separate DeepSeek provider profile.
Use a copied file selected by `OPENCODE_CONFIG` for the intended Docker process;
retain the existing Qwen profile and listener independently. This repository
does not edit a running container or the working Qwen installation.

The supplied handoff recommends OpenCode >=1.14.24. Configuration/provider
source was checked at OpenCode v1.19.3, commit
e3c4eb242585dcd77be1a867391b7c25fd47ce52. Its actual dependency is
`@ai-sdk/openai-compatible` 2.0.41. Source hashes and the SDK integrity pin are in
provenance/opencode-client.lock.json; transitive audit dependencies are locked.
The user's installed Docker version has not been inspected. Check that
container's `opencode --version` and effective provider/model selection after
the planned code reviews, before calling the actual client setup qualified.

The provider URL is `http://host.docker.internal:8081/v1`, model ID
`deepseek-v4.1-flash-1017-exl3`. Supply A's API key to the container through its
existing secure environment/secret mechanism as `DEEPSEEKRPC_API_KEY`. No key is
embedded in the example. Confirm Docker Desktop can reach A's configured
listener. If a private-LAN binding is necessary, select A's actual private IPv4,
configure its API-key file, and allow only required private access. An occupied
8081 is refused; no other listener is stopped.

Context capacity is 1,048,576. The profile's 8,192 output limit is separate and
shares that total input/output budget; it is not a 32,000 context cap or prefill
chunk size. Interleaved `reasoning_content`, image input and automatic function
tools are declared explicitly. Original thinking mode uses the original October
numeric reasoning-effort prompt when requested; this is not a guarantee of a
specific reasoning token count or a fabricated inference control. Unsupported
grammar/response-format/named-tool constraints fail clearly.

The profile leaves speculation off. After the real installed-cluster probe
passes on the current loaded components and mesh, change only this profile's
`speculative` option to true if desired. The server rejects an explicit true
request while that gate is unqualified. Full request timeout is disabled for
potentially long exact prefill; bounded header/chunk timeouts and cancellation
remain. The server sends heartbeat comments during model work.

The actual pinned SDK passed synthetic-wire checks for preserved text/reasoning
whitespace, split literal tool-call arguments, image data URLs, retained
tool-result/reasoning history and abort-signal propagation. Its two captured
actual request bodies also pass the original DeepSeek tokenizer/template/image
preparation with true 1M context and the requested 8,192 output limit. The
installed original vLLM parser has its separate preserving audit. These are
adapter/protocol checks, not a full Docker agent or GPU model session.

The later real-client qualification should cover a tool call and tool result,
multiple calls, Unicode/reasoning, an inline image, cancellation followed by a
new request, repeated turns, and compaction changing the prefix. Inspect actual
server reused/new/replay counts and client effective limits. Do not infer
correct context handling from a plausible short answer alone.
