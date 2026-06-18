# Metric Policy — WMH2017

## Initial required metrics

- validation loss
- Dice for WMH label 1

These are enough for smoke training only.

## Challenge metrics to add before serious comparison

- Dice
- 95th percentile Hausdorff distance
- average volume difference percentage
- lesion-wise recall
- lesion-wise F1

Individual lesions are 3D connected components in the challenge protocol.

## Do not overclaim from Dice

Dice alone does not adequately capture:

- small lesion misses
- false positive lesion count
- volume bias
- boundary distance
- scanner/site robustness

## Metric population labels

Every reported metric must specify:

- split: train / val / local heldout test
- metric implementation
- label policy
- case count
- scanner/site coverage
- run_id
- config hash
- dataset manifest hash
- split manifest hash

## Local heldout rule

If test labels exist in the 2022 local release and are evaluated, call the result `local heldout evaluation`.

Do not call it:

- official challenge score
- validation score
- customer-ready performance
- clinical performance
