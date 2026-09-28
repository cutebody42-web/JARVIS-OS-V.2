# Phase 2 — Sovereign Owner Kernel

Tracking: [issue #3](https://github.com/cutebody42-web/JARVIS-OS-V.2/issues/3), under [epic #1](https://github.com/cutebody42-web/JARVIS-OS-V.2/issues/1). Branch `nexus/phase-2-sovereign-owner-kernel`, stacked on `nexus/phase-1-provider-action-contracts` at `b7f2632bde53dd3cf27d61987fb07df2d3208512`. Draft only; no merge or deployment.

Phase 1's owner check protected the new executor but left Live and composite/background paths with ambient authority. Phase 2 adds a non-LLM owner kernel and a single authorization gateway in front of those entry points. Unknown actions, generated code, privilege escalation and unbounded legacy composites are denied. Models cannot provide policy, context, grants or approval tokens as tool arguments.

## Implementation and deliberate behavior changes

Typed frozen authority contracts, an explicit 34-capability registry, owner/session-bound consent, resource locks, strict normalized argument snapshots and v2 authorization receipts now sit between proposals and adapters. Exact consent expires within five minutes, is revocable and is atomically consumed before the effect. An adapter exception never restores the ticket.

Clock Reflex remains native and model-free. Workspace text creation/read can be exercised against real isolated POSIX files with verification evidence. Gmail API read/send, memory writes and search outputs remain UNVERIFIED when their legacy adapters only return prose. They cannot be converted into success merely by returning “Done.” Query disclosure and queuing a background model goal require consent too.

General browser/desktop/system, UI message sending, research/presentation composites and arbitrary runner callbacks are intentionally unavailable until their bounded operations are migrated. Their raw low-level builders were not rewritten. Tests that previously asserted ambient composite execution were updated to assert denial/no invocation; pure builder/render tests remain. Phase 1's legacy-prose test now supplies exact owner consent first, and its queue status test uses a native-read stub to preserve the original no-false-completion assertion. Provider swap, legacy receipt construction and model-free Reflex remain covered.

Authentication uses the existing API identity plus a fresh Live runtime. Review and execute exact requests through `/owner/*`; see [migration map](migration-map.md). Native confirmation UI, persistent/multi-process consent, complete queued-plan continuation, real Gmail delivery verification and Windows adapters remain deferred.

## Validation in Codex cloud

| Check | Result |
|---|---|
| Required application/security suite | **229/229 passed**, including 58 new Phase 2 tests |
| Minimal environment (`google-genai` plus stdlib) | **51/51 passed**: Phase 1 contracts/text routes + owner kernel |
| Full discovery, Phase 1 base | 242 run: 150 passed, 4 failures, 88 errors |
| Full discovery, Phase 2 | 322 run: 251 passed, 4 failures, 67 errors; **no new failed test IDs** |
| Existing staged QA | Compilation, dependency consistency, tracked-secret scan pass; full legacy tests and offscreen UI probe remain red |
| Frontend | Typecheck, lint and production build pass; UI files unchanged |
| Windows / actual devices | **Not run, not certified** |

Counts differ because headless imports now allow tests that previously failed collection. `main` no longer imports PortAudio before an actual hardware call, and optional PyAutoGUI import reports an absent display instead of crashing pure helper collection. This does not verify device health. One full-suite module (`test_action_helpers`) still fails import through the legacy screen processor's PortAudio dependency. Existing UI missing symbols/methods and four UI assertions are source/API failures, not relabelled as hardware failures. The offscreen probe still fails on missing `ui.INTRO_SEQUENCE_VERSION`.

Evidence: [cloud test result](phase-2-cloud-tests.json), [full baseline comparison](phase-2-full-audit.json), [benchmark](phase-2-benchmark.json). GitHub Actions evidence is linked in the draft PR after publication. `CI / backend`, `CI / frontend` and `NEXUS architecture contracts / offline-contracts` are the cloud review gates; repository branch-protection settings were not changed. The separate full legacy audit intentionally stays red while release blockers remain. No merge is permitted by this work.

Reproduce from the checkout in an isolated Python environment:

```sh
python -m unittest tests.test_nexus_contracts tests.test_nexus_text_route tests.test_owner_kernel -v
python scripts/cloud_validation.py --suite core --output artifacts/cloud-tests.json
python scripts/cloud_validation.py --suite full --output artifacts/full-tests.json
python scripts/qa.py automated
python scripts/benchmark_owner_kernel.py --baseline-dir ../phase1-checkout --output artifacts/benchmark.json
```

The runner uses an isolated SQLite DB unless an explicit test DATABASE_URL is supplied; GitHub's backend gate uses PostgreSQL with migrations. Offscreen/test doubles are code tests only. A future [Windows hardware gate](windows-hardware-gate.md) is preserved, closed and explicitly undispatched.

## Offline cost measurement

1,000 samples per path, same cloud Python runtime, real native clock/receipts and a stub provider with zero model/network latency. Median clock Reflex: Phase 1 **0.0373 ms**, Phase 2 **0.0684 ms** (about **0.0312 ms** extra). Model-planned clock median: **0.0626 → 0.1183 ms**. Provider calls remain **0** for Reflex and **1** for the stub-planned path; both return one receipt, now schema v2. These are noisy control-flow measurements, not model-quality, end-to-end voice, 32 GB RAM or integrated-GPU benchmarks.

## Review order and next steps

Read the [threat model](threat-model.md), [registry](capability-registry.md) and [migration map](migration-map.md), then review kernel ticket consumption, gateway dispatch and the adversarial tests. The SHA-256 digest binds equality; neither tickets nor receipts claim cryptographic tamper-proof auditing.

Next: native authenticated confirmation UX without changing the established visual design; durable protected receipts and shared consent coordination; exact external-service receipt verification; bounded Native/UIA/DOM capabilities with target locks; queue continuation/cancellation reconciliation; local-model policy and temporal memory; baseline UI repairs; deployable Windows build followed by supervised hardware certification. Broad legacy features must remain denied until migrated, not re-enabled by prompt text or blanket consent.
