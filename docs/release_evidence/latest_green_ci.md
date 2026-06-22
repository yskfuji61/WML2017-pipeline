# Latest green CI

Recorded for commit `d37e61355fcf30f8cc144e593968ad4103af0d83` (2026-06-22).

| Workflow | Trigger | Status | URL |
|----------|---------|--------|-----|
| structural-ci | push, pull_request | success | https://github.com/yskfuji61/WML2017-pipeline/actions/runs/27952801087 |
| security-scan | push, pull_request, weekly | success | https://github.com/yskfuji61/WML2017-pipeline/actions/runs/27952801059 |
| evidence-binder-ci | push, pull_request | success | https://github.com/yskfuji61/WML2017-pipeline/actions/runs/27952801058 |
| dependency-review | pull_request | not run on push | requires PR trigger |
| license-scan | push, pull_request | success | https://github.com/yskfuji61/WML2017-pipeline/actions/runs/27952801067 |
| release-candidate-ci | workflow_dispatch | manual | requires WMH2017_ROOT secret |

All workflows pin third-party Actions to full commit SHAs (see `.github/actions_pins.env`).

RE-WMH-006 updated to `pass` for structural/security/evidence/license scans at `d37e613`.
dependency-review and release-candidate-ci remain out of scope for this record.
