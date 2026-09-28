# Epic: Taby V2.0 / NEXUS

Tracking epic: [#1](https://github.com/cutebody42-web/JARVIS-OS-V.2/issues/1).
GitHub App authorization was repaired on 2026-09-28 with access limited to this
repository. Issues were enabled to host the requested planning epic.

## Goal and architecture

Preserve the current Taby UI. Build a local-first assistant for a 32 GB DDR5
laptop with integrated graphics:

Taby UI → Sovereign Owner Kernel → NEXUS Cognition → Model Gateway →
Native/UIA/DOM-first Action Kernel → Verification → temporal personal/world memory.

Reflex handles exact deterministic commands without model reasoning. Fast,
Standard and Mission routes are later bounded reasoning choices, with explicit
latency, resource and token budgets. Cloud models are optional providers rather
than the owner or execution authority. Do not load a large local model merely to
read a clock or perform another deterministic action.

Models must not modify security policy, replicate themselves, elevate privileges,
execute arbitrary code through unknown-tool fallbacks, or declare success without
trusted observations. Policy and action registration are application code reviewed
by the owner. Memory is evidence, not authority to override policy.

## Phase 1: architecture spike

Branch: `nexus/phase-1-provider-action-contracts`.
Base: `5fca0ac3e8cf4e9031072f9d27da8d83f6941e0e` (`main`).

- [x] Provider-neutral ModelProvider, with lazy Gemini text adapter.
- [x] Planner, executor and recovery decoupled from Gemini SDK types.
- [x] Structured ToolResult / ActionReceipt and evidence semantics.
- [x] Exact local clock Reflex route; no inference or network request.
- [x] Minimal immutable-by-interface owner policy and fail-closed dispatch.
- [x] Remove generated-code and unknown-tool execution fallbacks from executor.
- [x] Provider swapping, receipt, cancellation, refusal and text routing tests.
- [x] Offline baseline/new benchmark with declared measurement limits.
- [x] Existing QA/test suite run and baseline comparison recorded.
- [x] Create this main epic on GitHub.
- [ ] Publish the implementation and open a draft PR on the requested branch.
- [ ] Full required CI green and owner review before any merge.

## Next phases (planning only)

1. Reconcile existing QA failures without redesigning the UI. Validate Windows,
   audio, PostgreSQL and Redis on appropriate isolated runners.
2. Extend owner authorization across every action entry point, including Gemini
   Live/direct actions and specialized task runners. Define owner identity,
   capability grants, consent, revocation and threat model before expanding tools.
3. Add native API, UIA and DOM actions with schema validation and specific
   postcondition verifiers. Re-enable mutations individually, with rollback where
   possible. Coordinates and vision are fallback techniques.
4. Add bounded Fast/Standard/Mission routing and a real second provider adapter.
   Validate actual model latency and quality independently of local overhead.
5. Add temporal memory with provenance, observed/valid time, expiry, correction,
   contradiction handling, owner-controlled deletion and local encryption.
6. Measure Windows startup, idle/peak RAM, CPU and responsiveness on the target
   laptop. Budget concurrency and optional local model sizes from measurements;
   32 GB system RAM is not dedicated GPU VRAM.

## Phase 1 acceptance evidence

See [phase-1-spike.md](phase-1-spike.md), [validation.json](validation.json) and
[benchmark.json](benchmark.json). This is not a completed Sovereign Owner Kernel
or a security-certified/release-ready build. Keep the epic open after this spike.
