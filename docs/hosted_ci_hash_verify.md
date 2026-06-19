# Hosted CI artifact hash verification (GAP-014)

When `WMH2017_ROOT` is configured as a CI secret, `release_candidate_ci` may hash-verify real run artifacts locally in the hosted runner.

Without the secret, artifact hashes remain recorded in registers but hosted verification is `MISSING_WMH2017_ROOT`.

This is documentation-only until CI secret is approved by a human gate.
