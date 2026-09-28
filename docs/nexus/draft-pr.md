# Draft PR: NEXUS Phase 1 — provider-neutral cognition and evidence-bearing actions

Target: `main` from `nexus/phase-1-provider-action-contracts`.
Tracking epic: [#1](https://github.com/cutebody42-web/JARVIS-OS-V.2/issues/1).
Status: keep draft until required QA gates pass and the owner reviews the spike.

## Why

The existing planner/executor/recovery call Gemini directly. Unknown tools can
fall back to generated Python, and a normal string return can become a positive
model-generated completion summary. That couples reasoning, execution authority
and claims of success, preventing a safe local-first NEXUS architecture.

## Changes and resulting behavior

- Introduce neutral ModelProvider/ModelRequest/ModelResponse with a lazy,
  tenant-aware Gemini adapter. Swapping injected providers needs no planner or
  executor changes. No large model loads for a deterministic command.
- Validate bounded plans; fail closed on malformed output. Remove automatic code
  execution/unknown-tool fallbacks and model-generated completion summaries.
- Introduce ToolResult/ActionReceipt with typed evidence, immutable snapshots,
  status, IDs, timestamps and monotonic duration.
- Add an application-owned Phase 1 allowlist for clock/search/weather reads;
  all other agent-task capabilities are denied pending permission/verifier work.
- Add exact English/Arabic clock Reflex routing before planning and before a
  Live text turn. It works without a Live session and makes zero provider calls.
- Prevent TaskQueue from labeling denied/failed/unverified outcomes as completed;
  use a separate executor per task and expose copied receipts.
- Preserve `ui.py` and the entire `web/` tree. Add architecture/epic documentation,
  an offline benchmark and a separate contract CI workflow.

## Evidence

| Check | Base | Spike |
| --- | --- | --- |
| Full unittest runner | 212 reported: 120 pass, 4 fail, 88 errors | 242 reported: 150 pass, 4 fail, 88 errors |
| New offline contract/text tests | — | 30/30 pass, including fresh minimal venv |
| New failing test IDs | — | 0 |
| Source compilation / dependency check | pass | pass |
| Offscreen UI probe | fails: missing INTRO_SEQUENCE_VERSION | same failure |
| Frontend typecheck / lint / build | pass | identical frontend tree |

The full runner includes loader failures: missing PortAudio and DISPLAY prevent
some modules from collecting. Existing UI intro/tour symbols are also absent in
the base. The existing backend CI command was attempted and fails at the same
PortAudio import; PostgreSQL/Redis and remote GitHub CI are not certified.
Reproduction commands, baseline failure IDs and gate statuses are in
`docs/nexus/validation.json`. No merge while required gates remain red.

Offline benchmark, 500 samples/path, zero simulated model/network latency:

| Path | Median local ms | Provider calls/command | Receipts |
| --- | ---: | ---: | ---: |
| Old planner/executor (stubbed clock action) | 0.010 | 2 | 0 |
| New model path | 0.030 | 1 | 1 |
| New Reflex path | 0.020 | 0 | 1 |

This measures control-flow/receipt overhead, not live Gemini or laptop speed.
No end-to-end speedup, GPU/RAM claim or live-provider validation is implied.

## Limitations and next steps

This is a restricted architecture spike, not feature parity or a security
release. The owner boundary covers the migrated executor/new Reflex route only;
Gemini Live's other direct tools, specialized runners and legacy action internals
still need migration. Do not treat the whole application as secured.

Keep this PR draft. Restore the original required QA gates, extend owner policy
across all entry points, migrate native/UIA/DOM actions with real postcondition
verifiers, add a real second provider, then validate Windows/hardware and design
bounded temporal memory. The epic in `docs/nexus/epic.md` records these phases.
