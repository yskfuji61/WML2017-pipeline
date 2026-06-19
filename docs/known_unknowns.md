# Known unknowns

Items requiring human judgment or external confirmation before stronger release claims.

| id | topic | unknown | impact | mitigation |
|----|-------|---------|--------|------------|
| KU-001 | WMH2017 license | Final interpretation of CC-BY-NC boundaries for this use case | Blocks customer/commercial claims | REV-WMH-001 approved for research-only scope |
| KU-002 | Official evaluator license | `LICENSE_REVIEW.md` disposition is NOT_REVIEWED | Blocks official comparable metric claim | fetch fail-closed until APPROVED |
| KU-003 | Metric parity tolerance | Acceptable numeric delta vs official evaluator on fixture/real cases | Blocks leaderboard/SOTA claims | parity report fixture-only; claim blocked in release decision |
| KU-004 | Real E2E validity | Whether smoke training metrics reflect meaningful model quality | Blocks performance claims | model card limits; smoke-only wording |
| KU-005 | GitHub CI E2E | Hosted runners lack `WMH2017_ROOT`; full artifact hash gate not automatic on every PR | Preview promotion needs self-hosted or dispatch CI | release_candidate_ci + local preview-candidate |
| KU-006 | Dependency vulnerabilities | pip-audit reports known vulns in torch/monai stack | Security exceptions in register | exception register + periodic review |
| KU-007 | Subgroup fairness | Site/scanner subgroup performance not fully evaluated | External validity claims blocked | limitations documented in release decision conditions |
