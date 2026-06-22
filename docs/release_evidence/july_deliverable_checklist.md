# July deliverable checklist (slide-ready index)

**Purpose:** Single index for the July 2026 reporting format. Maps acceptance
criteria (AC), release checklist items, and current evidence paths. Not a
customer-facing deliverable.

**Recorded:** 2026-06-22 UTC  
**Commit:** `d37e61355fcf30f8cc144e593968ad4103af0d83`  
**Claim boundary:** Local validation only. Phase A/B gates **NOT met**. No
clinical, customer, proprietary-data, cloud, production, or SOTA claims.

---

## 1. Acceptance criteria (AC-01–AC-10)

| AC | Requirement | Status | Primary evidence |
|----|-------------|--------|------------------|
| AC-01 | WMH challenge data acquired | pass | [`docs/engineering_validation_plan.md`](../engineering_validation_plan.md); dataset audit scripts |
| AC-02 | Load images / labels | pass | `src/wmh2017/dataset/`; [`docs/experiment_journey.md`](../experiment_journey.md) Section E Phase 1 |
| AC-03 | Visualize ≥1 case | pass | [`artifacts/manifests/figure_manifest.json`](../../artifacts/manifests/figure_manifest.json) (case 100, RC2 fold0); local PNG under `reports/figures/overlays/` |
| AC-04 | Train/val split | pass | `data/splits/`; k-fold splits in Phase B1 |
| AC-05 | MONAI 3D training executes | pass | RC2 CV runs; [`registry/experiment_registry_wmh2017.csv`](../../registry/experiment_registry_wmh2017.csv) |
| AC-06 | Inference masks saved | pass | `artifacts/runs/wmh2017_rc2_cosine_fold*/` (local, gitignored weights) |
| AC-07 | Metrics calculated | pass | [`reports/cv/cv_summary_rc2_seed42.json`](../../reports/cv/cv_summary_rc2_seed42.json) |
| AC-08 | Experiment conditions recorded | pass | [`docs/experiment_journey.md`](../experiment_journey.md); run registry |
| AC-09 | Challenge comparison deferred | deferred | No official evaluator export; [`docs/release_evidence/official_evaluator_parity.md`](official_evaluator_parity.md) |
| AC-10 | No proprietary data / cloud | pass (offline) | REV-WMH-003 APPROVED; [`docs/security_privacy_gate.md`](../security_privacy_gate.md) |

Detailed AC pass table: [`docs/experiment_notes/wmh2017_local_e2e_seed42.md`](../experiment_notes/wmh2017_local_e2e_seed42.md).

---

## 2. RELEASE_CHECKLIST summary

Full checklist: [`RELEASE_CHECKLIST.md`](../../RELEASE_CHECKLIST.md).

| Area | Key items | Status |
|------|-----------|--------|
| Repository | Makefile, blocked claims in README | scaffold present |
| Reproducibility | run tree, materialized configs, hashes | preview E2E + RC2 runs (local artifacts) |
| Evaluation | case_metrics schema, test isolation, ADR-0007 selection | pass for preview + RC2 |
| CI / security | structural-ci, security-scan, evidence-binder, license-scan | **green at `d37e613`** — [`latest_green_ci.md`](latest_green_ci.md) |
| Human gates | license review, evaluator parity, reviewer sign-off | partial — see blocked section |

---

## 3. Latest evidence paths (8 reporting items)

| # | Item | Value / path |
|---|------|--------------|
| 1 | Commit hash / tag | `d37e613` — [`registry/package_identity_wmh2017.yaml`](../../registry/package_identity_wmh2017.yaml); tag null |
| 2 | Latest run_id | CV: `wmh2017_rc2_seed42`; reference fold0: `wmh2017_rc2_cosine_fold0_seed42` |
| 3 | Latest metrics | **Primary CV:** dice 0.612±0.047, recall 0.272±0.084, f1 0.354±0.056 — [`cv_summary_rc2_seed42.json`](../../reports/cv/cv_summary_rc2_seed42.json). **Reference fold0:** dice 0.671, recall 0.252 |
| 4 | Representative overlay | [`figure_manifest.json`](../../artifacts/manifests/figure_manifest.json); PNG local only |
| 5 | CI green URLs | [`latest_green_ci.md`](latest_green_ci.md) (RE-WMH-006 pass) |
| 6 | Official evaluator / license | [`official_evaluator_parity.md`](official_evaluator_parity.md) — NOT_FETCHED / NOT_REVIEWED; GAP-004/013 OPEN |
| 7 | This checklist | this file |
| 8 | Proprietary / cloud / clinical review | REV-WMH-003 offline APPROVED; FIND-WMH-008 formal gate NOT_REVIEWED; no dedicated clinical reviewer row |

---

## 4. Phase gate judgment (RC2)

| Gate | Threshold | RC2 CV | Judgment |
|------|-----------|--------|----------|
| Phase A Dice | 0.65 | 0.612 ± 0.047 | **NOT met** |
| Phase A Recall | 0.35 | 0.272 ± 0.084 | **NOT met** |
| Phase B Dice | 0.72 | 0.612 | **NOT met** |

Formal record: [`phase_gate_judgment_rc2.md`](phase_gate_judgment_rc2.md).  
State snapshot: [`docs/current_state_snapshot.md`](../current_state_snapshot.md).

---

## 5. Blocked claims (re-stated)

Do not claim:

- Clinical or diagnostic utility
- Customer presentation or production deployment
- Proprietary or cloud-processed data
- Official WMH challenge / leaderboard equivalence or SOTA
- Phase A or Phase B gate completion

Allowed: local CV metrics with variance, recall improvement vs A2-CV (+0.065),
and honest NOT-met gate status.
