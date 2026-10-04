# Development branch consolidation

Reviewed 4 October 2026. The owner requested that every development branch be
combined into `main`. The integration retains the commit history of all branch
tips below and uses the validated `product/jarvis-brain` implementation as its
baseline. Earlier competing runtimes are resolved to the implementation already
used by the packaged application; they are not installed alongside it as a second
brain or a second device trust system.

## Review decisions

The model profiler/router/runtime, event journal, semantic merge, synchronization,
owner kernel, durable missions and worker pool were compared with their earlier
branch versions. Current implementations retain the previous capabilities and
later fixes, including constrained-memory budgets, Ollama alias normalization,
owner-mediated actions, durable replay protection and revoked-peer removal.

Two compatible contributions are preserved explicitly:

- Ollama rejects blank replies at the provider boundary so the router can try the
  next local model. A loopback HTTP regression verifies successful local fallback.
- `core/improvement_pipeline.py` stores improvement proposals and per-check CI
  evidence in SQLite. This is a storage API, not a live autonomous improvement
  agent. It neither fetches CI evidence nor grants permission to install or merge;
  the existing repair/update coordinators keep their fresh checks and owner gates.

The current `PersonaAgentRuntime`/`RoutedModelProvider` stack remains selected.
Earlier separate persona context stores and unused streaming providers are
superseded. Shared JARVIS memory and bounded model transport remain in effect.
The current Ed25519 pairing/signing modules supersede the older HMAC device
registry and transport. The integrated scheduler has equivalent backoff/batching
and additionally removes revoked peers immediately.

All nine release gates now run for every `main` push. The release publisher binds
its Windows installer and Android companion to one exact source commit and
requires every gate to pass. `Install-JARVIS.cmd` provides a Windows source-folder
entrypoint for downloading and verifying that published installer.

## Reviewed tips

| Branch | Reviewed commit | Resolution |
| --- | --- | --- |
| `main` | `2904f34bee3e919ea684f84cd9a623fcf236fb28` | Already in product history; retained. |
| `nexus/fix-cloud-qa-regressions` | `60e7c9ade768ceab950b2dea343bb50b8b80ee4a` | Retain existing headless Qt, PostgreSQL health and import fixes plus expanded checks. |
| `nexus/integration-v2-runtime-sync` | `092725a5cba799962357583435a749c6f07daf7d` | Preserve independent improvement evidence ledger; retain current owner runtime and Ed25519 device transport. |
| `nexus/p0-agent-runtime` | `f887956a2a0e25bb0f6e1d87e0a6ac3e3cacee17` | Reconcile alternate implementations into the selected persona runtime and shared JARVIS memory. |
| `nexus/p0-event-journal` | `b1956cae14678de1eebd3147f78e44b5998a1acf` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-hardware-profile` | `78aa23b918bc54733c62fa9687fcafc6f080773b` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-merge-applier` | `2f9b884222c282b35f97be8eb0cbfaa606e0d116` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-merge-policy` | `ae5b8067113e27e436a3bf9c59cf949d7b3fa7c0` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-model-router` | `f737de419049d890149f90b776f716a23545765e` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-model-runtime` | `973f4e15fbcf8970ea7a280edf3d5a33a352d8e2` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-ollama-provider` | `a659d9055b95bbe41529b324a73617341f0c34bf` | Preserve blank-response rejection; retain current bounded local transport. |
| `nexus/p0-persona-runtime` | `605272f6bec22f1221cd051bd5b07ad2307b3b29` | Reconcile alternate implementations into the selected persona runtime and shared JARVIS memory. |
| `nexus/p0-persona-spec` | `e30e905a6af39a86e0bd479038c009528c2e17b2` | Reconcile alternate implementations into the selected persona runtime and shared JARVIS memory. |
| `nexus/p0-personas` | `88b2c108746be79d978edb6203057b76d78faa08` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p0-personas-runtime` | `746ce7ccd7e8b6168b94ebb5b1cbce0fbe93adef` | Reconcile alternate implementations into the selected persona runtime and shared JARVIS memory. |
| `nexus/p1-sync-daemon` | `965bda0baf1888ff394214f610fccf1864867a29` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/p1-sync-scheduler` | `6f47f937d77ec552029afcca12688702c79b1d49` | Retain integrated scheduler with batching, backoff and revoked-peer removal. |
| `nexus/p1-tailscale-transport` | `8944fb595f2c70c034a54156180d5f1eaf6378c5` | Retain current signed Ed25519 transport and persistent replay protection instead of the earlier HMAC transport. |
| `nexus/phase-1-provider-action-contracts` | `555a4d03e88d3c0eaeec786a96344fa1857ceadd` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/phase-2-sovereign-owner-kernel` | `aa28f1208443922a8f79f9c10267c2eb00e6d7e0` | Changes already represented; retain current implementation and subsequent fixes. |
| `nexus/phase-3-durable-mission-slice` | `e734f8f655a9183a2f87ba1721ac9ec04328491b` | Already in product history; retained. |
| `nexus/v2-finalization` | `dac723392a7587afe2de421b5a5013c283f9497a` | Retain integrated worker pool plus subsequent queued-cancellation capacity fix. |
| `nexus/v2-integration` | `94ffa0863a5feb76b1582fb0e9256eff30fa9bc7` | Already in product history; retained. |
| `product/jarvis-brain` | `e01f5a6439cc091e25046eed36199164df7e3e47` | Validated product implementation used as the integration baseline. |
