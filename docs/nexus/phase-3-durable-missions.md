# Phase 3 — durable action missions and local provider integration

Tracking [#5](https://github.com/cutebody42-web/JARVIS-OS-V.2/issues/5); stacked on draft #4 at `fffb487cc79a`. Draft only, no merge/deployment. The five design/research documents were committed before implementation (decision 003).

## Implemented slice

An opt-in local Ollama text provider implements the existing ModelProvider Protocol. It sends bounded `/api/chat` requests through a reused HTTP connection to a literal loopback origin. No proxy, redirect, model download, action dispatch or cloud fallback. A fake Ollama server tests the actual HTTP protocol end-to-end through the planner, kernel, native clock and persisted receipt. A real model/daemon was **not** installed or benchmarked.

`MissionService` accepts an explicit tool proposal or a goal that resolves to exactly one action. SQLite WAL with FULL synchronous commits records the exact request, digest, timestamps, bounded attempts, state events and last completed receipt. Native clock and exact-consent workspace read/create are admitted; all other capabilities remain unavailable on this new path. Multi-step plans are rejected wholesale. Existing Live/executor/queue authorization remains unchanged.

`state=succeeded` means **the admitted action** met its adapter postcondition. It is not semantic proof that an arbitrary natural-language goal was fulfilled: API responses explicitly include `completion_scope=single_action_only` and `goal_completion=not_evaluated`. `receipt_scope=last_completed_attempt` matters after a crash: an earlier confirmation receipt cannot explain the unrecorded later effect.

`VerifierResult` gives the receipt a typed pass/fail/unknown envelope with verifier identity and observed evidence. The gateway wraps the existing trusted adapter's proof; it does not claim a new independent universal verifier. Workspace creation uses exclusive/no-follow access, file fsync, readback/hash and parent-directory fsync. Legacy prose stays UNVERIFIED. Schema 1 receipt construction and schema 2 callers remain compatible.

## Owner API (no visual changes)

The service is disabled unless the host configures an absolute private `NEXUS_STATE_DIR` outside the application checkout. It does not require a Gemini Live session. Existing bearer authentication supplies the owner ID; request bodies cannot set owner, root, policy, capabilities or grants.

| Endpoint | Exact input / behavior |
|---|---|
| `POST /missions` | `{goal}`; strict Reflex first, otherwise explicitly configured local provider; create queued mission |
| `POST /missions/actions` | `{tool, arguments}`; normalize and snapshot without executing |
| `GET /missions` | Latest 50 owner-scoped metadata rows |
| `GET /missions/{id}` | Exact arguments/digest, fresh host session ID, state/events and receipt |
| `POST /missions/{id}/run` | `{session_id, version, ticket_id?}`; optimistic claim, kernel, verification, commit |
| `POST /missions/{id}/approve` | `{session_id, expected_digest, ttl_seconds?}`; only a currently pending kernel request |
| `POST /missions/{id}/revoke` | `{session_id, ticket_id}`; revocation through trusted owner channel |
| `POST /missions/{id}/cancel` | `{session_id, version}`; cancel queued/waiting or request in-flight cancellation |

For a safe demonstration: POST the goal `what time is it`, review the returned request, then POST run with its session/version. For a file: POST action `file_controller` with `{action: write, path: note.txt, content: exact text}`; first run returns waiting_confirmation without creating a file. Review its digest/arguments, approve, then run the **same mission** with the returned ticket and current version. No changed-arguments parameter is accepted on execution.

## Recovery and threat-model extension

- Persist RUNNING before any effect. If execution finishes but receipt commit fails, never report completion to the API; the journal remains RUNNING and restart marks it UNKNOWN. An OS-process-death test exercises this real gap after file creation. No automatic replay or compensation is claimed.
- A process-wide exclusive POSIX file lease prevents two live hosts from misclassifying each other's work as crashed. SQLite transactions and version checks prevent duplicate concurrent claims. Multi-process workers and Windows locking remain unsupported/fail closed.
- Tickets and live permissions are never serialized as reusable authority. Audit receipts may retain the ID of a consumed ticket. Restart creates a new kernel session; old approval/session is invalid. Waiting missions require a new run/review/approval. Terminal/unknown missions cannot be resumed; an owner must create a new explicit request after inspecting actual state.
- Cancellation cannot retract an already-started effect. An in-flight action may finish successfully with `cancel_requested=1`; the response does not falsely claim rollback.
- Model/web/email/file/memory text is inert. New routes are not registered as model tools. A plan cannot write policy, execute generated code, send externally or bypass workspace restrictions. These are capability/schema checks, not prompt-based security.
- Journal/source directories are outside action workspaces; owner queries are scoped in SQL. Private directory/no-follow checks reduce accidental exposure, but the store is not encrypted, signed or tamper-proof and is not a sandbox against the OS owner or hostile same-process Python. Disk-full/power-loss behavior cannot provide distributed exactly-once execution.
- Hard caps: 32 active/1,000 total missions per owner, 10,000 total per host, 128 live owner runtimes, 16 attempts/mission, 64 KiB arguments and 256 KiB receipt. No automatic retention deletion or pruning of owner history. Reaching quota blocks new work pending explicit maintenance.

## Validation and evidence

Cloud required suite: **263/263 passed**, including **34 new tests**. Phase 1 provider swaps, ActionReceipt behavior and model-free Reflex; Phase 2 consent expiry/reuse/revocation/injection/queue/API enforcement all remain in the required suite. New tests cover real loopback transport, provider selection, redirect/size/invalid-response rejection, tenant isolation, exact confirmation API, persistence, fresh consent after restart, concurrent claims, cancellation, unknown schema, limits, failed receipt commit and real process death after an effect.

Full discovery: **356 run / 285 pass / 4 failures / 67 errors**, versus Phase 2 **322 / 251 / 4 / 67**; no new failed test IDs. The full legacy workflow remains an explicit release blocker. Staged QA compilation/dependency/secret checks pass; legacy tests and offscreen UI probe retain their existing errors, including absent `INTRO_SEQUENCE_VERSION`. Windows/mic/camera/physical GPU checks are deferred, never simulated as passed.

See [cloud results](phase-3-cloud-tests.json), [full comparison](phase-3-full-audit.json), [500-sample benchmark](phase-3-benchmark.json). Minimal contracts run without full hardware dependencies; GitHub Actions links are attached to the draft PR at its published head. `ui.py` and `web/` have no changes; frontend CI still runs the existing checks.

Offline benchmark measures native clock and actual SQLite commits, not an LLM or laptop: compare Phase 2 bare Reflex, Phase 3 bare Reflex, and Phase 3 create/run/persist/read. All paths make zero model calls; the durable case has ten warmups and verifies persisted success count after reopening. See JSON for machine-specific median/p95 rather than treating targets as promises.

## Limits and next slice

This is a synchronous single-action mission foundation, not a full DAG engine, durable migration of the old queue, streaming voice provider, real-device build or autonomous personal knowledge system. Planning before mission creation is not checkpointed. HTTP socket timeout does not impose a hard total deadline against a trickling server. Real Ollama compatibility/model quality/offline server configuration and actual Windows performance need separate certification. Existing Gemini Live audio and visual UI remain intact.

Next: repair legacy release blockers; preserve visual design while adding owner review; migrate one old queue workflow onto durable steps; add shared cancellation/deadlines and an independently verified external receipt; certify local inference/Windows packaging; then evaluate the optional memory/UIA/voice adapters against the documented budgets. Prediction and reflex compilation remain disabled R&D. No automatic merge.
