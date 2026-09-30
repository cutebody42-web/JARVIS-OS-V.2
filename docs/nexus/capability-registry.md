# Phase 2 capability registry

Generated from `core/capability_registry.py`. Code metadata is the policy source; this inventory is documentation, never loaded as authority. Unknown tools/sub-actions are denied. All 28 Live declarations are mapped; `system_time` is also exposed to the Reflex/executor path.

`external` includes query disclosure to search providers. `disabled` capabilities cannot be enabled by a ticket. A verifier requirement means unverified adapter prose cannot claim success.

| Capability | Access / risk | External / sensitive | Destruction / privilege | Reversible | Confirmation | Surface / lock | Available |
|---|---|---|---|---|---|---|---|
| system.time | read / RiskLevel.LOW | False / False | DestructiveLevel.NONE / user | True | none | ExecutionSurface.NATIVE / none | Admitted adapter |
| web.search | read / RiskLevel.MODERATE | True / False | DestructiveLevel.NONE / user | False | exact | ExecutionSurface.NETWORK / none | Admitted adapter |
| weather.read | read / RiskLevel.MODERATE | True / False | DestructiveLevel.NONE / user | False | exact | ExecutionSurface.NETWORK / none | Admitted adapter |
| workspace.read | read / RiskLevel.MODERATE | False / True | DestructiveLevel.NONE / user | True | exact | ExecutionSurface.NATIVE / workspace | Admitted adapter |
| workspace.create_text | mutate / RiskLevel.HIGH | False / False | DestructiveLevel.NONE / user | True | exact | ExecutionSurface.NATIVE / workspace | Admitted adapter |
| email.send | mutate / RiskLevel.HIGH | True / True | DestructiveLevel.IRREVERSIBLE / user | False | exact | ExecutionSurface.NETWORK / email | Admitted adapter |
| email.read | read / RiskLevel.MODERATE | False / True | DestructiveLevel.NONE / user | True | exact | ExecutionSurface.NETWORK / email | Admitted adapter |
| memory.write | mutate / RiskLevel.MODERATE | False / True | DestructiveLevel.RECOVERABLE / user | True | exact | ExecutionSurface.INTERNAL / memory | Admitted adapter |
| task.submit | mutate / RiskLevel.MODERATE | True / False | DestructiveLevel.NONE / user | True | exact | ExecutionSurface.INTERNAL / none | Admitted adapter |
| task.status | read / RiskLevel.LOW | False / True | DestructiveLevel.NONE / user | True | exact | ExecutionSurface.INTERNAL / none | Admitted adapter |
| task.cancel | mutate / RiskLevel.MODERATE | False / False | DestructiveLevel.RECOVERABLE / user | False | exact | ExecutionSurface.INTERNAL / none | Admitted adapter |
| desktop.open_app | mutate / RiskLevel.CRITICAL | False / False | DestructiveLevel.NONE / user | True | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| message.read_legacy | mutate / RiskLevel.CRITICAL | False / True | DestructiveLevel.NONE / user | True | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| message.draft_legacy | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.DOM / desktop | Denied legacy bundle |
| message.send | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.DOM / desktop | Denied legacy bundle |
| reminder.schedule | mutate / RiskLevel.CRITICAL | False / True | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.INTERNAL / timers | Denied legacy bundle |
| media.youtube | mutate / RiskLevel.CRITICAL | True / False | DestructiveLevel.NONE / user | True | forbidden | ExecutionSurface.DOM / desktop | Denied legacy bundle |
| media.control | mutate / RiskLevel.CRITICAL | True / False | DestructiveLevel.NONE / user | True | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| sensors.vision | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.NONE / user | False | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| system.settings | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / admin | False | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| browser.legacy | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.DOM / desktop | Denied legacy bundle |
| desktop.legacy | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| code.generated | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.NATIVE / workspace | Denied legacy bundle |
| code.developer_agent | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.NATIVE / workspace | Denied legacy bundle |
| desktop.input | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| software.updater | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / admin | False | forbidden | ExecutionSurface.NATIVE / workspace | Denied legacy bundle |
| travel.legacy | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.NONE / user | True | forbidden | ExecutionSurface.NETWORK / desktop | Denied legacy bundle |
| ui.graphics | mutate / RiskLevel.CRITICAL | False / False | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| ui.control | mutate / RiskLevel.CRITICAL | False / False | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.DESKTOP / desktop | Denied legacy bundle |
| research.composite | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.NETWORK / workspace | Denied legacy bundle |
| presentation.composite | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.NATIVE / workspace | Denied legacy bundle |
| file.processor_legacy | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.NATIVE / workspace | Denied legacy bundle |
| system.self_shutdown | mutate / RiskLevel.CRITICAL | False / False | DestructiveLevel.RECOVERABLE / user | True | forbidden | ExecutionSurface.INTERNAL / desktop | Denied legacy bundle |
| worker.arbitrary_callable | mutate / RiskLevel.CRITICAL | True / True | DestructiveLevel.IRREVERSIBLE / user | False | forbidden | ExecutionSurface.NATIVE / workspace | Denied legacy bundle |

The JSON inventory includes tool aliases, environment metadata, verifier requirements and disabled reasons. Gmail read/send require existing owner-scoped credentials; connection setup is not a model capability. Workspace adapters use POSIX no-follow directory handles and fail on unsupported Windows platforms. No Windows filesystem certification is claimed.
