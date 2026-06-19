# Review gate matrix (v4)

| Gate | Required evidence | Required reviewer | Blocks next phase |
|---|---|---|---|
| First data load | dataset manifest, header DLP audit | engineering reviewer | yes |
| First smoke training | run log, loss, prediction manifest | engineering reviewer | yes |
| First metric report | metric artifact, metric tests | engineering reviewer | yes |
| Official comparison | source register, metric protocol, parity test | evidence reviewer | yes |
| Preview | evidence binder, security scan, CI log, reviewer approval | engineering reviewer | yes |
| Release | not applicable in this repository | release owner | blocked |

All reviewers default to `PENDING_CONFIRMATION` until human assignment.
