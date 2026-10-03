import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/jarvis-runtime.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const mediaType = "application/vnd.nexus-sync+json";

function runtime(invoke, biometric, fetchImplementation = () => { throw new Error("Mobile transport must stay native"); }) {
  const exports = {};
  const context = {
    exports,
    Response,
    URL,
    TextEncoder,
    atob,
    btoa,
    fetch: fetchImplementation,
    window: { __TAURI__: { core: { invoke }, biometric }, setTimeout },
  };
  vm.runInNewContext(compiled, context, { filename: "jarvis-runtime.ts" });
  return exports;
}

test("mobile chat reaches tailnet HTTP through Rust and verifies its signed response", async () => {
  const commands = [];
  const api = runtime(async (command, args) => {
    commands.push(command);
    if (command === "mobile_sign_brain_request") {
      return { request_id: "chat-1", endpoint: "http://100.90.10.3:8765/nexus/brain/v1/message", body: "signed-request" };
    }
    if (command === "mobile_companion_post") {
      assert.equal(args.offer, null);
      assert.equal(args.body, "signed-request");
      assert.match(args.endpoint, /^http:\/\/100\./);
      return { status: 200, content_type: mediaType, body: "signed-response" };
    }
    if (command === "mobile_verify_brain_response") {
      assert.equal(args.requestId, "chat-1");
      assert.equal(args.signedResponse, "signed-response");
      return { identity: "JARVIS", text: "ready", lane: "local" };
    }
    throw new Error(command);
  });
  const reply = await api.mobileBrainMessage("hello");
  assert.equal(reply.text, "ready");
  assert.deepEqual(commands, ["mobile_sign_brain_request", "mobile_companion_post", "mobile_verify_brain_response"]);
});

test("mobile approval list and decision use native transport and native biometric signing", async () => {
  const commands = [];
  const api = runtime(async (command, args) => {
    commands.push(command);
    if (command === "mobile_sign_approval_list" || command === "mobile_sign_approval_decision") {
      if (command === "mobile_sign_approval_decision") {
        assert.equal(args.approvalId, "approval-1");
        assert.equal(args.approved, true);
      }
      return { request_id: "approval-request", endpoint: "http://100.90.10.3:8765/nexus/approval/v1/pending", body: "signed-request" };
    }
    if (command === "mobile_companion_post") {
      return { status: 200, content_type: mediaType, body: "signed-response" };
    }
    if (command === "mobile_verify_approval_list_response") return { pending: [{ approval_id: "approval-1" }] };
    if (command === "mobile_verify_approval_receipt") return { state: "approved" };
    throw new Error(command);
  });
  const pending = await api.mobilePendingApprovals();
  assert.equal(pending[0].approval_id, "approval-1");
  assert.equal((await api.mobileDecideApproval("approval-1", true)).state, "approved");
  assert.equal(commands.filter((command) => command === "mobile_companion_post").length, 2);
  assert.ok(commands.includes("mobile_sign_approval_decision"));
});

test("mobile pairing sends only invitation-scoped requests and accepts signed desktop approval", async () => {
  const offer = { inviter_endpoint: "http://100.90.10.3:8765", expires_at: new Date(Date.now() + 60000).toISOString() };
  let posts = 0;
  let authentications = 0;
  const api = runtime(async (command, args) => {
    if (command === "mobile_fingerprint_status") {
      return { available: true, fingerprint: true, biometryType: 1, error: null };
    }
    if (command === "mobile_verify_owner_presence") {
      assert.match(args.reason, /pairing/);
      authentications += 1;
      return;
    }
    if (command === "mobile_prepare_pairing") {
      return { request: { pairing_id: "pair-1" }, submit_url: `${offer.inviter_endpoint}/nexus/pair/v1/request`, status_url: `${offer.inviter_endpoint}/nexus/pair/v1/status` };
    }
    if (command === "mobile_companion_post") {
      assert.equal(args.offer, offer);
      posts += 1;
      return posts === 1
        ? { status: 202, content_type: "application/json", body: "{}" }
        : { status: 200, content_type: mediaType, body: "signed-pairing" };
    }
    if (command === "mobile_accept_pairing_approval") {
      assert.equal(args.signedApproval, "signed-pairing");
      return { device_id: "mobile-1" };
    }
    throw new Error(command);
  });
  assert.equal((await api.pairMobileCompanion(offer)).device_id, "mobile-1");
  assert.equal(posts, 2);
  assert.equal(authentications, 1);
});

test("desktop error detail survives native mobile transport", async () => {
  const api = runtime(async (command) => {
    if (command === "mobile_sign_approval_list") {
      return { request_id: "list", endpoint: "http://100.90.10.3:8765/nexus/approval/v1/pending", body: "request" };
    }
    return { status: 403, content_type: "application/json", body: '{"detail":"Companion is not approved"}' };
  });
  await assert.rejects(api.mobilePendingApprovals(), /Companion is not approved/);
});

test("pairing cannot use face-only availability or proceed after native fingerprint cancellation", async () => {
  let preparations = 0;
  const faceOnly = runtime(async (command) => {
    if (command === "mobile_fingerprint_status") {
      return { available: false, fingerprint: false, biometryType: 0, error: "Enroll a fingerprint first" };
    }
    preparations += 1;
    throw new Error(command);
  });
  await assert.rejects(faceOnly.pairMobileCompanion({}), /Enroll a fingerprint/);
  assert.equal(preparations, 0);

  const cancelled = runtime(async (command) => {
    if (command === "mobile_fingerprint_status") return { available: true, fingerprint: true, biometryType: 1, error: null };
    if (command === "mobile_verify_owner_presence") throw new Error("Fingerprint was cancelled");
    preparations += 1;
    throw new Error(command);
  });
  await assert.rejects(cancelled.pairMobileCompanion({}), /Fingerprint was cancelled/);
  assert.equal(preparations, 0);
});

test("a cancelled native fingerprint approval sends no decision to the desktop", async () => {
  const commands = [];
  const api = runtime(async (command) => {
    commands.push(command);
    if (command === "mobile_sign_approval_decision") {
      throw new Error("Fingerprint verification failed or was cancelled");
    }
    throw new Error("No decision may leave this phone without fingerprint proof");
  });
  await assert.rejects(api.mobileDecideApproval("approval-1", true), /Fingerprint verification failed/);
  assert.deepEqual(commands, ["mobile_sign_approval_decision"]);
});

test("desktop updates use authenticated explicit steps and bind installation to checkpoint and approval", async () => {
  const requests = [];
  const status = {
    supported: true, phase: "awaiting_approval", message: "Verify on your phone",
    plan: null, checkpoint_id: "checkpoint-1", approval_id: "approval-1",
    approval_state: "pending", history: [],
  };
  const api = runtime(async () => { throw new Error("Checking or preparing cannot close JARVIS"); }, undefined,
    async (url, init) => {
      requests.push({ url, init });
      return new Response(JSON.stringify(status), { status: 200, headers: { "Content-Type": "application/json" } });
    });
  const client = new api.LocalBrainClient({ endpoint: "http://127.0.0.1:8765", token: "ui-token" });
  assert.equal((await client.getUpdateStatus()).approval_state, "pending");
  await client.checkUpdates();
  await client.prepareUpdate();
  assert.equal(requests.length, 3);
  assert.ok(requests.every(({ url }) => !url.endsWith("/apply")));
  await client.applyUpdate("checkpoint-1", "approval-1");
  assert.deepEqual(requests.map(({ url }) => new URL(url).pathname), [
    "/v1/update/status", "/v1/update/check", "/v1/update/prepare", "/v1/update/apply",
  ]);
  for (const { init } of requests) assert.equal(init.headers.Authorization, "Bearer ui-token");
  assert.equal(requests[1].init.method, "POST");
  assert.equal(requests[2].init.method, "POST");
  assert.deepEqual(JSON.parse(requests[3].init.body), { checkpoint_id: "checkpoint-1", approval_id: "approval-1" });
});

test("desktop update shutdown is an explicit native command", async () => {
  const commands = [];
  const api = runtime(async (command) => { commands.push(command); });
  assert.deepEqual(commands, []);
  await api.shutdownForUpdate();
  assert.deepEqual(commands, ["shutdown_for_update"]);
});
