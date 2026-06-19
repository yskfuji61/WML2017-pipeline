# Security and privacy gate

## Scope

The current package may handle public WMH2017 challenge files locally after
source/terms review. It must not process proprietary data, patient-identifiable
data, customer data, or cloud-uploaded data without prior written approval.

## DLP classification

| Artifact class | DLP status | Rule |
|---|---|---|
| Source code and docs | Public/Internal scaffold | may be stored in repo |
| Public WMH2017 images | Public dataset with license boundary | local only; do not redistribute unless terms allow |
| Generated manifests | Internal evidence | may be committed only if no raw patient identifiers or absolute private paths |
| Generated predictions/checkpoints | Internal research artifact | do not publish or customer-share before review |
| Proprietary images | Restricted/regulated | blocked until Security/Privacy + Dataset Governance approval |
| Credentials/API keys | Restricted | never include |

## Stop conditions

- Any patient identifier, credential, API key, employee identifier, or private customer identifier appears.
- Cloud execution, upload, external API call, or shared drive transfer becomes necessary.
- Dataset terms prohibit the intended action.
- Logs contain absolute private paths or sensitive metadata that cannot be masked.
- A reviewer cannot determine whether a file is public, internal, confidential, or restricted.

## DLP rule (v4)

No proprietary/private/PHI/PII data use, storage, model training, export, upload, or report inclusion.
Local redacted metadata inspection for DLP/security review is allowed only to determine whether review is required.
Raw metadata values must not be printed, committed, logged, exported, or included in reports.
If PHI/PII-like metadata is found, stop and require security/privacy review.
