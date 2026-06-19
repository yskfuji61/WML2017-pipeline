# Official metric parity report

## Conclusion

Local metric computation matches the official evaluator on the synthetic empty-mask fixture. Official comparable metrics are **not approved** because the evaluator source is not fetched and license review is pending.

## Evidence

- Report: `reports/evaluation/official_metric_parity_report.json`
- Verifier: `python scripts/verify_official_metric_parity.py`
- Supply-chain gate: `python scripts/verify_official_evaluator_source.py` (EXPECTED_FAIL until fetch + license review)

## Fixture result summary

| fixture_id | metric | local | official | delta | pass |
|------------|--------|-------|----------|-------|------|
| synthetic_empty_mask | dice | 1.0 | 1.0 | 0.0 | true |

## Blocked claims

Leaderboard equivalence, SOTA claims, and "official comparable" wording remain blocked (`claims_allowed.official_comparable: false`).
