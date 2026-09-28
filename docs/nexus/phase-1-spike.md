# Phase 1 architecture spike: provider and action contracts

## Decision

Planning depends on `ModelProvider.generate(ModelRequest) -> ModelResponse`.
Requests contain neutral text, system instructions, a tier and JSON-output intent.
Gemini translates them in `core/providers/gemini.py`; SDK imports, credentials,
model IDs, explicit client lifetime and 30-second request timeout stay there.
Credentials resolve per request through existing tenant-aware configuration.
No process-global provider/client or SDK key configuration is introduced.

The planner validates a bounded JSON plan (at most five independent steps,
unique IDs, parameter objects and known metadata). Unknown metadata, forged
receipts, malformed responses and provider failures produce no executable plan.
The legacy `_fallback_plan` helper remains for existing callers/tests but is no
longer invoked automatically after model failure.

The executor obtains a plan, passes every proposed action through application
policy, captures a receipt, and renders outcomes deterministically. Replanning
uses the same injected provider and has a two-replan cap. Denial, unverified
output, cancellation and Reflex errors never trigger model recovery. The old
code-generation/unknown-tool execution fallback and model-generated success
summary are removed. Error recovery helpers use the same neutral interface.

## Action boundary and compatibility

The Phase 1 executor admits only `system_time` with no parameters,
`web_search` with a bounded query, and `weather_report` with a bounded city.
**Other agent-task capabilities are intentionally denied in this spike.** They
need explicit owner grants and trustworthy verifiers before they can return.
This is a behavior restriction, not a drop-in parity release.

Policy is a code-owned allowlist with strict parameter keys. Plans cannot set,
replace or bypass it. QA policy is an additional check, not a substitute for
owner policy. There is no shell, generated-code handler, policy mutation tool,
privilege escalation tool or self-replication route in this executor.

Legacy search/weather adapters still contain their own provider dependencies;
this spike decouples cognition, not every existing action implementation. Their
string returns are `unverified`, even when the text says `Done`. They are not
retried or described as completed. The clock handler creates a `succeeded`
result backed by the exact system-clock observation it returned. This verifies
that local read; it does not prove external clock accuracy or a user's goal.

`ToolResult` distinguishes succeeded/failed/denied/cancelled/unverified. Success
requires a nonempty tuple of typed evidence records. `ActionReceipt` includes
schema version, task/action/step IDs, a JSON parameter snapshot, route, timezone
aware timestamps, monotonic elapsed duration and result. Parameters and evidence
are snapshotted/immutable; exported JSON does not mutate a receipt. Evidence is
created by trusted action code, never copied from model plan metadata. This is
an in-process contract, not a cryptographic attestation or OS sandbox.

TaskQueue now uses an executor per task to avoid concurrent receipt/provider
state contamination. Non-successful agent outcomes do not become `completed` or
receive the old unconditional completion announcement. Public task status
includes copied receipts. There is no new persistent receipt store or automatic
export of receipt data. Existing task-history persistence is unchanged.

## Reflex integration and UI

Exact normalized phrases include `What time is it?`, `time`, `الساعة كام؟` and
`الساعه كام`. Whitespace/case/terminal question punctuation are normalized.
Compound commands, timezone questions and injected suffixes do not match.

The route runs before model initialization in both planning and execution. Typed
commands enter it before sending a Gemini Live text turn, even without a Live
session. Startup/shutdown gates still block dispatch. Existing log/subtitle
methods display the local result. `ui.py` and every `web/` source/config file are
byte-for-byte unchanged from the base. If hosted, the returned local timestamp
is the runtime/server clock with its offset, not a claim about the user's phone.

Voice recognition/Gemini Live session creation still use Gemini. This spike
does not introduce offline speech recognition, a new UI, a local large model,
UIA/DOM action implementations, or temporal memory storage.

## Verification and benchmark

See `validation.json` for the observed environment, counts, baseline failure IDs,
commands and gate state. New offline tests cover independent provider injection,
SDK-free cognition imports, Gemini request translation/lifetime, malformed plans,
failed/cancelled/denied/unverified results, evidence requirements, parameter
snapshots, concurrency, refusal with no recovery, bounded replanning and actual
Live text methods with only an inert audio import substitute.

Run the new independent CI gate:

```sh
python -m venv .venv-nexus
# Activate the virtual environment for the platform, then:
python -m pip install google-genai
python -m unittest -v tests.test_nexus_contracts tests.test_nexus_text_route
```

Run existing QA in an isolated environment after installing `requirements.txt`:

```sh
JARVIS_QA_MODE=1 QT_QPA_PLATFORM=offscreen JARVIS_ENV=test \
  DATABASE_URL=sqlite:////absolute/path/to/isolated-test.db \
  python scripts/qa.py automated
```

Use PostgreSQL/Redis for the existing hosted CI gate; local SQLite results do
not certify those integrations. Linux desktop tests also require PortAudio and
a display/Xvfb. The existing offscreen probe independently fails because the
base UI lacks `INTRO_SEQUENCE_VERSION`; no UI fix is hidden in this spike.

Reproduce the offline benchmark (20 warmups, 500 samples per path):

```sh
git worktree add --detach ../nexus-baseline 5fca0ac3e8cf4e9031072f9d27da8d83f6941e0e
python scripts/benchmark_nexus.py --baseline-dir ../nexus-baseline \
  --iterations 500 --output docs/nexus/benchmark.json
```

The benchmark runs the old planner/executor code with an inert Gemini module and
stubbed clock action; the new paths run the real native clock/receipt code with
a stub text provider. No legacy generated code, network or desktop action runs.
Observed calls/command: old = 2 (plan + summary), new model = 1, new Reflex = 0.
With zero network/model delay, median local costs were approximately 0.010 ms,
0.030 ms and 0.020 ms respectively. Receipt overhead costs more than the old
stubbed path locally; no end-to-end speedup is claimed. Exact observations and
p95s are in `benchmark.json`. Live latency, output quality, Windows behavior,
audio, startup RAM and integrated-GPU performance were not benchmarked.

## Release blockers and next steps

- The existing full suite is not green on the base or the spike; preserve its
  failure list and require appropriate CI plus owner review before merging.
- The narrow owner policy covers AgentExecutor and the new typed Reflex route.
  **Gemini Live direct tool dispatch, specialized runners and existing action
  modules are not migrated.** They retain legacy capabilities and risks. Do not
  claim this application as a whole prevents policy modification, replication,
  privilege escalation or unsupported success claims. Gate/migrate those paths
  before enabling broader autonomy or treating this as a security boundary.
- A production second provider, streaming/multimodal interfaces, per-action
  verifiers, rollback, durable/redacted audit retention, owner identity and
  grants, temporal memory, and resource budgets remain future phases.
- GitHub App authorization was repaired on 2026-09-28 for this repository only.
  The planning epic is [#1](https://github.com/cutebody42-web/JARVIS-OS-V.2/issues/1).
  Remote CI evidence must be recorded separately from the local baseline tests.
