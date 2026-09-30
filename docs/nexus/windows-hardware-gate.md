# Future Windows/hardware certification gate

**Status: blocked / not certified.** Phase 2 was built and tested in Codex cloud and GitHub Actions. The future 32 GB DDR5 Windows laptop, larger SSD and integrated GPU have not been accessed or measured. No microphone, speaker, camera or desktop success is inferred from a stub or offscreen Qt test.

The manual `NEXUS Windows hardware certification (deferred)` workflow is reserved for an explicitly provisioned physical self-hosted Windows runner labelled `nexus-hardware`. It is not dispatched during this spike and currently fails closed. It is a preserved release gate, not an implemented certification harness. Hosted Windows VMs must not be relabelled as the target laptop.

After a deployable build exists, review and implement a supervised test harness before opening this gate. Record the exact build SHA, OS/build, device/driver identities, RAM/storage configuration, operator approval, measured evidence and limitations for each item:

1. Packaged installation, startup, cleanup and owner/session identity binding.
2. Workspace containment against Windows reparse points, junctions, hardlinks, alternate data streams and source-policy paths; crash recovery of partial artifacts.
3. Exact consent/revocation across native/UIA/DOM actions, stable target identity, cancellation and verifier observations.
4. Microphone permission denial/acceptance, actual capture, speaker playback and interruption, with retained observations and no hidden mocks.
5. Camera permission denial/acceptance and real frame capture only with the owner's explicit consent.
6. UIA/DOM-first desktop interactions in a controlled test app, privilege boundaries and denied UAC/admin actions.
7. CPU, RAM, model loading, disk usage, thermal behavior and end-to-end latency/soak on the actual 32 GB machine; report integrated-GPU support honestly.
8. Existing UI regressions fixed and full cloud QA green before any release claim.

A human approval or checkmark without device observations is not hardware verification. Policy/source changes to open the gate require review; no model output, memory entry or action ticket can modify it.
