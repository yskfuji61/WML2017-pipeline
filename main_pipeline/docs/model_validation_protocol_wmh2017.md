# Model validation protocol — WMH2017 local smoke phase

## Scope

This protocol covers local technical validation only. It does not validate
clinical performance, patient-level decision making, diagnostic safety, customer
presentation, or production use.

## Validation levels

| Level | Name | Evidence | Allowed claim |
|---|---|---|---|
| MV-0 | structure only | files exist, unit tests pass | scaffold exists |
| MV-1 | data intake | dataset manifest, label audit, split manifest | public data can be read locally |
| MV-2 | smoke training | train log, checkpoint hash, prediction mask | pipeline executes |
| MV-3 | local metric | local Dice/case metrics linked to run evidence | local validation result only |
| MV-4 | protocol parity | official evaluator parity, source review, reviewer disposition | comparable under stated deviations |
| MV-5 | external/clinical | independent clinical/model validation review | outside current scope |

## Required metric linkage

Every metric row must link to:

- run_id
- model/config version
- code commit or package hash
- dataset manifest hash
- split manifest hash
- prediction artifact hash
- metric script hash
- metric definition
- spacing/orientation handling note
- reviewer disposition
- claim boundary

## Metric limitations

Dice alone is insufficient for small multifocal WMH behavior. Before any serious
comparison, include lesion-wise F1/recall and surface/distance or AVD measures
as applicable to the reviewed protocol. Local metric output must not be called
official benchmark output unless official evaluator parity is documented.
