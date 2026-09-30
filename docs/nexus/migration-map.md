# Phase 2 action migration map

All 28 current Live declarations map to explicit capabilities. Registration does not imply permission: unsafe legacy bundles are deliberately disabled until their constituent operations have bounded arguments and verifiers. The visual files `ui.py` and `web/` are unchanged.

| Entry / path | Boundary now | Behavior / remaining work |
|---|---|---|
| `JarvisLive._execute_tool` and batches | `owner_runtime.gateway.run_tool` | No pre-gateway dispatch, transcript approval, mutable default path injection, dynamic risk-name lists or sensor thread. Every call returns a v2 receipt. Batches preserve proposal order. |
| `main._lazy_action` compatibility aliases | Gateway before any legacy module import | A direct compatibility call cannot restore the old ambient dispatcher. |
| `AgentExecutor` → `run_action` | Same captured runtime and gateway | Provider-neutral plan proposals; no replan on denial/confirmation/unverified result. Clock Reflex has zero model calls. |
| `TaskQueue.submit` → worker executor | Captures owner runtime, restores tenant and passes runtime into executor | Queuing a model task through Live needs exact approval; each worker step is independently authorized. Workers cannot enqueue workers. |
| `TaskQueue.submit_action` | Checks the exact request/ticket when the worker executes | Revocation, expiry and cancellation while queued are respected. Trusted host API, not a model tool. |
| `TaskQueue.submit_job` arbitrary runners | `worker.arbitrary_callable` is denied | Runner and its arbitrary callbacks are never invoked. Reintroduce specialized work only as explicit bounded capabilities. |
| Task status/cancel | Owner-scoped lookup; separate `task.status` and `task.cancel` capabilities | Confirmation required. Single-action owner execution does not automatically resume or complete the original multi-step task; awaiting tasks require host cancellation/reconciliation. |
| `email_control` public entry | Decorated gateway entry; exact Gmail read/send adapters call private bounded helpers | Bare approve/global draft cannot send. No browser/Apple Mail fallback. Existing credentials required. Returned prose stays UNVERIFIED until provider receipt verification is implemented. |
| `send_message`, `prepare_message_reply`, browser-draft helpers | Guarded, disabled DOM/message capability | Mutable focus/recipient state cannot be ticketed safely yet. No message send occurs, even when text claims approval. |
| `file_controller` | Relative workspace read / exclusive text create adapter | Exact approval, no-follow handles and readback hashes. Legacy open/delete/move/rename/arbitrary paths denied. Windows adapter deferred. |
| `file_processor`, `code_helper`, `dev_agent` | Guarded forbidden capabilities | Generated code, arbitrary document transformations and source edits unavailable. |
| `browser_control` | Guarded forbidden composite | General navigation/evaluate/click/type disabled; future Native/UIA/DOM primitives need explicit targets/verifiers. |
| `deep_research`, request/queue wrappers | Guarded forbidden composite | Existing search/synthesis/build helpers remain trusted low-level code. No automatic browser, file or nested-runner access. |
| Presentation create/request/queue/legacy wrappers | Guarded forbidden composite | Pure builders remain unit tested; model entry cannot launch them. Future migration must separately authorize inputs, egress and artifacts. |
| Open app, settings, desktop/input, updater | Guarded forbidden capabilities | No power/admin/install/privilege escalation. Hardware certification and narrow adapters needed. |
| Screen processor, media, messages, reminder, flight, graphics/UI model tools | Explicit disabled mappings; public action wrappers guarded where present | Device/UI/session-wide actions are unavailable to models in this spike. Owner-facing visual UI remains unchanged. |
| Web/weather | Approved query-only adapter | Query disclosure itself requires confirmation. Weather no longer opens a browser. Raw search implementation is trusted adapter code, not a grant or arbitrary URL executor. |
| Memory save | Exact `memory.write`; category/key/value schema | Memory stays data. Policy/config categories forbidden; durable write verification deferred. |
| `/chat` and WebSocket text | Text/model transport, no owner approval handler | Chat “yes” does not mint tickets. Authenticated `/owner/*` routes are separate and not tool declarations. |
| Native UI callbacks / low-level Python helpers | Trusted application host boundary | This phase is not an OS sandbox. These must not become dynamic model-callable entry points. Audio hardware imports in `main` are deferred; actual audio use can still fail. |

## Reviewable owner API flow

With an authenticated active Live session, request `GET /owner/actions/pending`. Review the returned capability, exact normalized arguments, digest and session ID. Submit `POST /owner/actions/{request_id}/approve` with `session_id`, `expected_digest` and optional `ttl_seconds` (1–300). The response contains a ticket, not action success.

To invoke that unchanged snapshot, submit `POST /owner/actions/{request_id}/execute` with `session_id` and `ticket_id`. New arguments, owner IDs and capability overrides are rejected. `DELETE /owner/consents/{ticket_id}?session_id=...` revokes unused consent; `DELETE /owner/actions/{request_id}?session_id=...` cancels the pending request. Reconnect/shutdown invalidates the old runtime. The model has no HTTP/owner-control tool for these routes.

There is intentionally no visual confirmation UI in this phase. This is a reviewable backend contract, exercised by authenticated HTTP tests. Native desktop owner-channel integration and multi-worker shared state are next steps.
