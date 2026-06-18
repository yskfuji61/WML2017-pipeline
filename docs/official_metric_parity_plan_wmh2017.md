# Official metric parity plan

## Purpose

Prevent local Dice or convenience metrics from being misrepresented as official
WMH challenge, leaderboard, SOTA, or reproduced results.

## Current state

```text
official_evaluator_located: false
official_evaluator_hash_recorded: false
official_metric_parity_completed: false
leaderboard_or_sota_claim_allowed: false
```

## Required parity record

Before any official-comparable claim, record:

- official evaluator source_id
- exact URL/DOI/repository
- version/date
- sha256 of evaluator code or release archive
- license/terms
- expected input format
- expected label convention
- spacing/orientation assumptions
- metrics implemented
- local wrapper command
- fixture input
- fixture expected output
- local output
- deviation list
- reviewer
- disposition

## Local metric boundary

`src/wmh2017/evaluation/**` and `scripts/evaluate_wmh2017.py` may support local
engineering validation. They do not establish official benchmark equivalence.
