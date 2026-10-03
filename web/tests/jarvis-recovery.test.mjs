import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/jarvis-runtime.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const old = { endpoint: "http://127.0.0.1:10001", token: "old-token" };
const fresh = { endpoint: "http://127.0.0.1:10002", token: "fresh-token" };
const json = (value, status = 200) => new Response(JSON.stringify(value), { status });

function runtime(invoke, fetch) {
  const exports = {};
  vm.runInNewContext(compiled, {
    exports, fetch, Response, URL, TextEncoder, AbortSignal, Error,
    window: { __TAURI__: { core: { invoke } }, setTimeout },
  });
  return exports;
}

test("a failed read reconnects and retries once with the new sidecar token", async () => {
  const requests = [];
  let starts = 0;
  const api = runtime(async (command) => {
    assert.equal(command, "ensure_brain_sidecar");
    starts += 1;
    return fresh;
  }, async (url, init) => {
    requests.push({ url, token: init.headers?.Authorization });
    if (url.startsWith(old.endpoint)) throw new Error("Sidecar exited");
    if (url.endsWith("/health")) return json({ ok: true, identity: "JARVIS" });
    return json({ identity: "JARVIS", brain_ready: true });
  });
  const client = new api.LocalBrainClient(old);
  assert.equal((await client.status()).brain_ready, true);
  assert.equal(starts, 1);
  assert.equal(client.connection, fresh);
  assert.equal(requests.filter(({ url }) => url.endsWith("/status")).length, 2);
  assert.equal(requests.at(-1).token, "Bearer fresh-token");
});

test("concurrent failed reads share one native reconnect and health wait", async () => {
  let starts = 0;
  const api = runtime(async () => {
    starts += 1;
    await new Promise((resolve) => setTimeout(resolve, 5));
    return fresh;
  }, async (url) => {
    if (url.startsWith(old.endpoint)) throw new Error("Sidecar exited");
    return json(url.endsWith("/health") ? { ok: true, identity: "JARVIS" } : { identity: "JARVIS" });
  });
  const client = new api.LocalBrainClient(old);
  await Promise.all([client.status(), client.status()]);
  assert.equal(starts, 1);
});

test("a chat with a lost response is never replayed or automatically restarted", async () => {
  let posts = 0;
  let starts = 0;
  const api = runtime(async () => { starts += 1; return fresh; }, async (url, init) => {
    assert.equal(init.method, "POST");
    assert.ok(url.endsWith("/message"));
    posts += 1;
    throw new Error("Response lost after possible delivery");
  });
  await assert.rejects(new api.LocalBrainClient(old).message("Open the app"), /possible delivery/);
  assert.equal(posts, 1);
  assert.equal(starts, 0);
});

test("a server error does not kill or replace an alive brain", async () => {
  let starts = 0;
  const api = runtime(async () => { starts += 1; return fresh; }, async () => json({ detail: "Busy" }, 503));
  await assert.rejects(new api.LocalBrainClient(old).status(), /Busy/);
  assert.equal(starts, 0);
});

test("a stale read token gets one retry but a stale POST token gets none", async () => {
  let starts = 0;
  let posts = 0;
  const api = runtime(async () => { starts += 1; return fresh; }, async (url, init) => {
    if (init.method === "POST") { posts += 1; return json({ detail: "Bad token" }, 401); }
    if (url.startsWith(old.endpoint)) return json({ detail: "Old token" }, 401);
    return json(url.endsWith("/health") ? { ok: true, identity: "JARVIS" } : { identity: "JARVIS" });
  });
  const client = new api.LocalBrainClient(old);
  await client.status();
  await assert.rejects(client.message("Approve operation"), /Bad token/);
  assert.equal(starts, 1);
  assert.equal(posts, 1);
});

test("read recovery has at most one replay even when the new endpoint rejects authentication", async () => {
  let reads = 0;
  const api = runtime(async () => fresh, async (url) => {
    if (url.endsWith("/health")) return json({ ok: true, identity: "JARVIS" });
    reads += 1;
    if (reads === 1) throw new Error("Old endpoint gone");
    return json({ detail: "Replacement token rejected" }, 401);
  });
  await assert.rejects(new api.LocalBrainClient(old).status(), /Replacement token rejected/);
  assert.equal(reads, 2);
});

test("an accepted update handoff suspends reconnect while waiting for shutdown", async () => {
  let starts = 0;
  const api = runtime(async () => { starts += 1; return fresh; }, async (url) => {
    if (url.endsWith("/apply")) return json({ phase: "applying" });
    throw new Error("Brain shutting down");
  });
  const client = new api.LocalBrainClient(old);
  await client.applyUpdate("checkpoint", "approval");
  await assert.rejects(client.status(), /shutting down/);
  assert.equal(starts, 0);
});

test("an ambiguous update response cannot start another sidecar or replay installation", async () => {
  let starts = 0;
  let installations = 0;
  const api = runtime(async () => { starts += 1; return fresh; }, async (url) => {
    if (url.endsWith("/apply")) installations += 1;
    throw new Error("Connection lost");
  });
  const client = new api.LocalBrainClient(old);
  await assert.rejects(client.applyUpdate("checkpoint", "approval"), /Connection lost/);
  await assert.rejects(client.status(), /Connection lost/);
  assert.equal(installations, 1);
  assert.equal(starts, 0);
});

test("an explicit rejected update leaves ordinary read recovery available", async () => {
  let starts = 0;
  const api = runtime(async () => { starts += 1; return fresh; }, async (url) => {
    if (url.endsWith("/apply")) return json({ detail: "Owner approval expired" }, 403);
    if (url.startsWith(old.endpoint)) throw new Error("Old sidecar exited later");
    return json(url.endsWith("/health") ? { ok: true, identity: "JARVIS" } : { identity: "JARVIS" });
  });
  const client = new api.LocalBrainClient(old);
  await assert.rejects(client.applyUpdate("checkpoint", "approval"), /expired/);
  await client.status();
  assert.equal(starts, 1);
});
