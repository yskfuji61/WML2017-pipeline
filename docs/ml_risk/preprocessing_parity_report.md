# Preprocessing parity report

## Conclusion

Training and evaluation share config materialization from the same YAML config path. Geometry and resampling policies are bound to config SHA256 recorded in run evidence.

## Evidence

- Config: `configs/wmh2017_monai_smoke.yaml`
- Materialization: E2E stages `train_smoke_model_stage` / `evaluate_stage`
- Run evidence fields: `config_path`, `config_sha256` in `registry/run_evidence_schema_wmh2017.csv`

## Known gaps

- Full cross-run preprocessing diff automation is not implemented (manual review for preview promotion).
- Additional annotation masks (`additional_annotations/**`) are excluded from smoke baseline per split policy.

## Blocked claims

No claim of clinical-grade preprocessing validation or multi-site deployment readiness.
