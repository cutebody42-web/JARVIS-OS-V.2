# Cloud development and certification

Development, builds, tests, research and benchmarks run in Codex cloud and GitHub Actions. The owner currently has no development laptop available. Future Windows / 32 GB DDR5 / integrated graphics is a target, never a measured environment.

| Gate | What it proves | Current execution |
|---|---|---|
| Unit | Pure policy, state transitions, parsing and deterministic routes | Cloud, required |
| Contract | Provider swaps, consent binding, evidence/receipt shape | Cloud, required |
| Cloud integration | Real isolated SQLite/files/loopback HTTP; auth and migrations | Cloud, required |
| Full legacy QA | Existing behavior and release blockers remain visible | Cloud, failures retained |
| Windows certification | Packaged startup, UIA, path/reparse safety, native cancellation | **DEFERRED** |
| Physical hardware | Mic/camera, wake word, voice latency, iGPU/RAM/thermal behavior | **DEFERRED** |

Fakes are explicitly named transport/adaptor doubles. They do not certify an Ollama model, Windows desktop, audio, camera or GPU. Missing PortAudio/display libraries are environment-dependent where the traceback establishes that cause; missing UI methods and failed assertions are code failures. Do not blanket-skip old failures or label them hardware issues.

Use isolated environments/worktrees, no access to the owner's local files. Required checks: `scripts/cloud_validation.py --suite core`; `nexus-contracts.yml`; existing frontend lint/typecheck/build. Full `--suite full` and `scripts/qa.py automated` stay visible in the separate release-blocker workflow. Add new test modules to the explicit cloud list. Run migrations with PostgreSQL in the backend CI as before. Keep UI files byte-for-byte unchanged in architecture-only PRs.

The Phase 3 mission service is opt-in with `NEXUS_STATE_DIR` (a private, persistent POSIX directory outside model workspaces). One API process owns its SQLite journal; a second process refuses to open it. Hosted production defaults still require PostgreSQL for existing accounts/chats; the opt-in mission journal is a separate local-first prototype, **not a supported multi-replica hosted store**. Do not enable it behind multiple Uvicorn workers. Backups need a quiescent process or SQLite backup API; encryption/retention/export administration is deferred.

For local model transport set `NEXUS_MODEL_PROVIDER=ollama`, `NEXUS_OLLAMA_MODEL` to an already installed, owner-reviewed model, and optionally `NEXUS_OLLAMA_URL=http://127.0.0.1:11434`. There is no model download or daemon installation in CI. Existing unspecified provider behavior remains Gemini for compatibility. The new goal-mission route uses an explicitly configured local provider and never silently sends the goal to cloud.

Future certification must use a deployable build on an explicitly authorized Windows device: cold/warm launch, UIA read/write and postcondition, reparse points, DPI/multimonitor changes, permission prompts, pause/cancel, crash recovery, wake false accepts/rejects, English/Arabic recognition, offline egress capture, model cold/warm RSS and p95 response, thermal load and battery. Publish machine/build/model hashes, raw observations and failures. The existing manual Windows gate stays closed until that harness exists; this work does not dispatch it or pretend it passed.

Publish small commits and stacked **draft** PRs. New required failures must be fixed before presenting the slice as passing. Unresolved baseline release failures forbid merge/deployment. Nothing in this work modifies branch protection or automatically merges drafts.

## Full QA repair after Phase 3

The historical failures in the Phase 2/3 evidence remain accurate for those commits.
The stacked `nexus/fix-cloud-qa-regressions` repair is documented in
[ADR 004: contract migration](qa-contract-migration.md) and
[measured repair evidence](qa-regression-evidence.json). It restores cloud collection,
fixes shipped startup/configuration defects, and explicitly replaces unsupported
UI specifications. Full local discovery: **367/367, zero skips**; minimal contracts:
**79/79**. The full legacy workflow still fails on real errors and now also requires
the cloud checklist and installed localhost Chromium. Hardware remains deferred.

The physical Windows gate is unchanged; green cloud CI is not release/hardware
certification. Broad exception handling and hard-coded theme colors remain static
P2 review observations, not failed executable checks. No new model/dependency
integration or visible UI redesign is part of this repair.
