# Latest green CI

| Workflow | Trigger | Status | URL |
|----------|---------|--------|-----|
| structural-ci | push, pull_request | PENDING | PENDING |
| security-scan | push, pull_request, weekly | PENDING | PENDING |
| evidence-binder-ci | push, pull_request | PENDING | PENDING |
| dependency-review | pull_request | PENDING | PENDING |
| license-scan | push, pull_request | PENDING | PENDING |
| release-candidate-ci | workflow_dispatch | manual | requires WMH2017_ROOT secret |

Record actual GitHub Actions run URLs after the next green run. Until then RE-WMH-006 remains `blocked` in the evidence register.

All workflows pin third-party Actions to full commit SHAs (see `.github/actions_pins.env`).
