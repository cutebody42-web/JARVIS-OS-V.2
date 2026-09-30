# NEXUS performance budget

These are engineering targets and admission limits, not laptop benchmark claims. Cloud measurements live in each phase's evidence JSON. Phase 2's 1,000-sample clock Reflex median was 0.0684 ms, p95 0.1277 ms; stub-planned clock median 0.1183 ms. This excludes network/model inference and physical devices.

| Path | Initial target | Guard / measurement |
|---|---|---|
| Reflex routing + kernel | p95 < 5 ms warm, excluding tool | Zero provider calls, deterministic strict match |
| Durable native read mission | p95 < 25 ms warm on cloud local disk | Includes SQLite FULL sync; benchmark separately from bare Reflex |
| FAST | First useful response < 1 s on future target | Provisional, requires actual small-model benchmark |
| STANDARD | First useful response < 3 s warm | Provisional; stream later, cap context and output |
| MISSION | Immediate queued acknowledgement, bounded progress | No autonomous multi-worker execution yet |
| Perception | Event-triggered, one observation per relevant change | No per-frame VLM loop |

Prototype bounds: one action/mission, at most five planner proposals but accept exactly one, 64 KiB canonical action arguments, 16 action attempts per mission, 32 active and 1,000 total missions per owner, bounded list responses. Ollama: one reused connection per provider, one in-flight request per instance, 60 s socket timeout, 64 KiB request and 256 KiB response bounds, 4,096 context tokens and 512 output-token option, five-minute model keep-alive. Socket timeout is not a hard end-to-end deadline against a trickling server; process cancellation of inference is future work.

Future 32 GB planning envelope (not reservation/measurement): OS/apps 10 GB; one quantized primary model plus KV cache up to 12 GB; voice/embeddings at most 3 GB; assistant/browser working set 3 GB; 4 GB headroom. Start CPU-first, test iGPU only after driver certification. Never keep multiple large models resident without measured headroom. Fail/defer an expensive mission rather than swapping the machine into unusability.

Keep Reflex out of the inference semaphore. Cache small immutable schemas and bounded retrieval; reuse model sessions; progressively expose tools; compress history with provenance. Parallelize independent reads only after worker budgets/locks exist. Do not benchmark framework README numbers as NEXUS performance. Record CPU/runtime, sample count, warmup, medians/p95, provider calls and limits. No claim that cloud disk or loopback stub represents future laptop or real model speed.
