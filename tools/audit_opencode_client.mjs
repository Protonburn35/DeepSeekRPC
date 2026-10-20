// Actual pinned AI SDK adapter; synthetic wire fixtures, no model/OpenCode process.
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { deflateSync } from "node:zlib";

const args = process.argv.slice(2);
const get = name => args[args.indexOf(name) + 1];
assert(args.includes("--modules") && args.includes("--output"), "explicit modules and new output directories are required");
const modules = path.resolve(get("--modules")), output = path.resolve(get("--output"));
const require = createRequire(import.meta.url);
const sdk = require(path.join(modules, "@ai-sdk/openai-compatible"));
const metadata = JSON.parse(fs.readFileSync(path.join(modules, "@ai-sdk/openai-compatible/package.json")));
assert.equal(metadata.version, "2.0.41");
const pins = JSON.parse(fs.readFileSync(new URL("../provenance/opencode-client.lock.json", import.meta.url)));
const adapterSHA = crypto.createHash("sha256").update(fs.readFileSync(path.join(modules, "@ai-sdk/openai-compatible/dist/index.js"))).digest("hex");
assert.equal(adapterSHA, pins.sdk.dist_index_js_sha256);
fs.mkdirSync(output, { recursive: false });
const requests = [], checks = [];
const modelId = "deepseek-v4.1-flash-1017-exl3";
const make = chunks => sdk.createOpenAICompatible({ name: "deepseekrpc", baseURL: "http://127.0.0.1:1/v1", includeUsage: true,
  fetch: async (url, init) => {
    assert.equal(String(url), "http://127.0.0.1:1/v1/chat/completions");
    requests.push(JSON.parse(init.body));
    return new Response(chunks.map(value => "data: " + JSON.stringify(value) + "\n\n").join("") + "data: [DONE]\n\n",
      { status: 200, headers: { "content-type": "text/event-stream" } });
  }
}).chatModel(modelId);
const chunk = (delta, finish_reason = null) => ({ id: "chatcmpl-fixture", created: 0, model: modelId,
  object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason }] });
const usage = { id: "chatcmpl-fixture", created: 0, model: modelId, object: "chat.completion.chunk", choices: [],
  usage: { prompt_tokens: 20, completion_tokens: 7, total_tokens: 27 } };
const controls = { maxOutputTokens: 8192, seed: 41017, temperature: .7,
  providerOptions: { deepseekrpc: { thinking: { type: "enabled" }, speculative: false } } };
const text = make([chunk({ role: "assistant" }), chunk({ reasoning_content: " \nreason " }),
  chunk({ content: " \nanswer  " }), chunk({}, "stop"), usage]);
const response = await text.doStream({ ...controls, prompt: [{ role: "user", content: [{ type: "text", text: "fixture request" }] }] });
const events = [];
for await (const event of response.stream) events.push(event);
assert.equal(events.filter(e => e.type === "reasoning-delta").map(e => e.delta).join(""), " \nreason ");
assert.equal(events.filter(e => e.type === "text-delta").map(e => e.delta).join(""), " \nanswer  ");
assert(events.some(e => e.type === "finish"));
assert.deepEqual(requests[0].thinking, { type: "enabled" });
assert.equal(requests[0].max_tokens, 8192);
checks.push("actual SDK request controls, preserving text/reasoning deltas and usage completion");

// Deterministic valid PNG from independent ordinary RGB pixels.
const crc32 = buffer => {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
};
const pngChunk = (type, data) => {
  const size = Buffer.alloc(4), crc = Buffer.alloc(4), payload = Buffer.concat([Buffer.from(type), data]);
  size.writeUInt32BE(data.length); crc.writeUInt32BE(crc32(payload));
  return Buffer.concat([size, payload, crc]);
};
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(32, 0); ihdr.writeUInt32BE(32, 4); ihdr[8] = 8; ihdr[9] = 2;
const pixels = Buffer.alloc(32 * (1 + 32 * 3));
for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
  const start = y * 97 + 1 + x * 3, value = (x + y) % 2 * 255;
  pixels[start] = value; pixels[start + 1] = 255 - value; pixels[start + 2] = value;
}
const png = Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), pngChunk("IHDR", ihdr),
  pngChunk("IDAT", deflateSync(pixels)), pngChunk("IEND", Buffer.alloc(0))]);
const toolArgs = '{"value":"true","literal":"<arguments>{x}</arguments>","n":4}';
const toolModel = make([chunk({ role: "assistant" }),
  chunk({ tool_calls: [{ index: 0, id: "call_fixture", type: "function", function: { name: "echo", arguments: toolArgs.slice(0, 11) } }] }),
  chunk({ tool_calls: [{ index: 0, function: { arguments: toolArgs.slice(11) } }] }), chunk({}, "tool_calls"), usage]);
const toolResponse = await toolModel.doStream({ ...controls, tools: [{ type: "function", name: "echo", description: "Fixture tool",
  inputSchema: { type: "object", properties: { value: { type: "string" } } } }], toolChoice: { type: "auto" },
  prompt: [{ role: "user", content: [{ type: "text", text: "Inspect fixture" }, { type: "file", mediaType: "image/png", data: png }] },
    { role: "assistant", content: [{ type: "reasoning", text: "retained original reasoning" },
      { type: "tool-call", toolCallId: "call_previous", toolName: "echo", input: { value: "old" } }] },
    { role: "tool", content: [{ type: "tool-result", toolCallId: "call_previous", toolName: "echo", output: { type: "text", value: "original tool result" } }] },
    { role: "user", content: [{ type: "text", text: "Continue" }] }] });
const toolEvents = [];
for await (const event of toolResponse.stream) toolEvents.push(event);
assert.equal(toolEvents.find(e => e.type === "tool-call").input, toolArgs);
assert(requests[1].messages[0].content[1].image_url.url.startsWith("data:image/png;base64,"));
assert.equal(requests[1].messages[1].reasoning_content, "retained original reasoning");
assert.equal(requests[1].messages[2].tool_call_id, "call_previous");
checks.push("actual SDK image data URL, tool-result/reasoning input and split literal tool-call argument preservation");

const controller = new AbortController();
const cancellable = sdk.createOpenAICompatible({ name: "deepseekrpc", baseURL: "http://127.0.0.1:1/v1",
  fetch: async (_url, init) => new Promise((_resolve, reject) => {
    init.signal.addEventListener("abort", () => reject(new DOMException("fixture cancellation", "AbortError")), { once: true });
  }) }).chatModel(modelId);
const cancelled = cancellable.doStream({ ...controls, abortSignal: controller.signal,
  prompt: [{ role: "user", content: [{ type: "text", text: "fixture" }] }] });
setTimeout(() => controller.abort(), 10);
await assert.rejects(cancelled, error => error.name === "AbortError");
checks.push("actual SDK abort signal reaches its fetch owner");
fs.writeFileSync(path.join(output, "requests.json"), JSON.stringify(requests, null, 2) + "\n");
const report = { scope: "actual installed AI SDK with synthetic wire fixtures; no OpenCode process or GPU model execution",
  sdk: metadata.version, checks: checks.map(name => ({ name, status: "passed" })),
  adapter_sha256: adapterSHA };
fs.writeFileSync(path.join(output, "sdk.json"), JSON.stringify(report, null, 2) + "\n");
process.stdout.write(JSON.stringify(report, null, 2) + "\n");
