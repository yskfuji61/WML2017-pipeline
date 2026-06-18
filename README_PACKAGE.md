# WMH2017 Pipeline — local public-data PoC scaffold

> package_version: `0.2.3`
>
> local planning state: `READY_FOR_REQUIREMENTS_REVIEW`
>
> controlled release state: `NOT_READY_FOR_PREVIEW`
>
> This package is a June catch-up scaffold for loading, splitting, visualizing,
> MONAI 3D smoke training, inference, and local Dice evaluation on public
> WMH2017 data.
>
> Real WMH2017 run evidence, source/license review, official metric parity,
> review/approval, and release decision are not complete. Clinical use,
> customer presentation, proprietary-data processing, cloud execution,
> production deployment, and SOTA/leaderboard claims are blocked.
>
> Audit entry points: [`README_CURSOR_START.md`](README_CURSOR_START.md),
> [`docs/release_state_crosswalk.md`](docs/release_state_crosswalk.md), and
> [`docs/final_evidence_binder_index.md`](docs/final_evidence_binder_index.md).


> Canonical implementation entry: [`README_CURSOR_START.md`](README_CURSOR_START.md).
>
> Current release_state: `READY_FOR_REQUIREMENTS_REVIEW`.
>
> Structural checks, real-data runs, preview readiness, and release readiness are separate states.
> This package is not preview-ready, not release-ready, and not approved for clinical or customer use.

> Start here for implementation work: [`README_CURSOR_START.md`](README_CURSOR_START.md).
>
> Current release_state: `READY_FOR_REQUIREMENTS_REVIEW`.
>
> This package now includes a lightweight MONAI smoke-pipeline scaffold under `src/wmh2017/`.
> Legacy ISLES-derived code remains under `core/pipeline/` and should not be treated as the active WMH2017 baseline until audited.

---

# wmh2017-reproducible-pipeline

**Language:** English | [Japanese](README_ja.md)

Reproducible segmentation pipeline for the **MICCAI 2017 White Matter Hyperintensities
(WMH) Segmentation Challenge** (Kuijf et al., IEEE TMI 2019), bootstrapped from the
sibling repo [`isles2022-2d3d-blend-reproducible-pipeline`](https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline)
(heterogeneous 2D + 3D + 2.5D ensemble for ISLES 2022) and re-targeted for
**FLAIR + T1** input.

This 0.2.3 release is the **inheritance baseline**: the code is in place but
WMH-specific bits are marked with `# DEFERRED_WMH_REVIEW:` headers. WMH-data activation
is the explicit Phase 1 work item; see [ROADMAP.md](ROADMAP.md).

**Quick links**
- Entry guide (EN): [wmh2017/README_en.md](wmh2017/README_en.md)
- Entry guide (JA): [wmh2017/README.md](wmh2017/README.md)
- **Inheritance map** (what came from where): [docs/inheritance/inheritance_map.md](docs/inheritance/inheritance_map.md)
- **Experiment journey** (lessons from ISLES + WMH plan): [docs/experiment_journey.md](docs/experiment_journey.md)
- Audit map: [AUDIT_MAP.md](AUDIT_MAP.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Citation: [CITATION.cff](CITATION.cff)

## What this repository provides today (0.2.3)

- A **portfolio-grade repo skeleton** matching the sibling
  `yskfuji/*-reproducible-pipeline` convention.
- The **MPS-compatible nnU-Net 3D trainer** (Apple Silicon `ConvTranspose3d`
  workaround) inherited verbatim — this is the single most reusable piece of
  the ISLES work for any 3D MRI segmentation task on Apple hardware.
- **MONAI smoke MPS compatibility patch** (`src/wmh2017/training/mps_compat.py`):
  when MPS is selected, `ConvTranspose3d` is replaced with nearest upsample +
  `Conv3d` for smoke/compatibility testing only; numerical equivalence with the
  original architecture is not claimed.
- The **cross-architecture probability fusion script** with case-adaptive
  thresholding (`core/pipeline/scripts/cross_arch_ensemble_native.py`). It
  will be used in Phase 2 once WMH models exist; until then the script is
  task-agnostic and ready.
- The **2.5D ConvNeXt model** (`ConvNeXtNnUNetSeg`) and its training loop,
  with WMH calibration UNRESOLVED_PLACEHOLDER markers (channel counts, slice offsets,
  pos_slice_weight).
- A **no-data smoke test** that exercises the inherited model + the adaptive
  threshold logic with synthetic volumes in < 30 seconds.

## What this repository does NOT yet provide

- A trained WMH model. The first WMH model is a Phase 1 deliverable.
- WMH-tuned configs. The ISLES configs were deliberately **not** copied.
- WMH-specific dataset I/O. `wmh_dataset.py` is a renamed copy of
  `isles_dataset.py` with DEFERRED_WMH_REVIEW markers for the FLAIR / T1 / mask paths,
  scanner-stratified CSV columns, and "other pathology" class exclusion.
- The MICCAI 5-metric evaluation suite (HD95, AVD, lesion F1, Recall). The
  evaluator currently reports Dice only; extending it is in ROADMAP Phase 1.

## Phase-1 acceptance criteria (per kickoff brief)

The 6月 catch-up phase has 10 acceptance criteria; this repo supplies the
skeleton for AC-02 / AC-05 / AC-06 / AC-07. AC-01 (data acquisition) and
AC-08 / AC-09 / AC-10 (documentation and review) are author responsibilities.

| AC | Description | Inherited support | WMH-specific work needed |
|---|---|---|---|
| AC-01 | Obtain WMH data | n/a | Download from https://wmh.isi.uu.nl/, DUA |
| AC-02 | Load FLAIR + label | `wmh_dataset.py` skeleton | Fill DEFERRED_WMH_REVIEW stubs |
| AC-03 | Visualize ≥ 1 case | n/a (manual) | nibabel / matplotlib |
| AC-04 | Train/val split | `pandas` in `wmh_dataset.py` | Scanner-stratified CSV |
| AC-05 | Train baseline model | `train_wmh_25d_convnext.py` OR MONAI 3D U-Net | Use MONAI for baseline (per brief) |
| AC-06 | Inference masks | `evaluate_wmh_25d.py` | minor adaptation |
| AC-07 | Dice metric | `metrics_segmentation.py` | extend to 5-metric MICCAI suite |
| AC-08 | Lab notebook | Methodology in `docs/experiment_journey.md` | author writes notebook |
| AC-09 | Compare to published | Phase 4 "reality check" in journey doc | author analysis |
| AC-10 | No proprietary data leaks | `.gitignore` excludes `Datasets/`, weights, logs | author discipline |

## Quickstart

### 1. Verify the inherited pipeline without WMH data

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. Inspect what's in the public bundle

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. Read the inheritance map before touching code

```bash
less docs/inheritance/inheritance_map.md
```

### 4. Start Phase 1 with the MONAI baseline

Per the kickoff brief, AC-05 calls for a MONAI / PyTorch standard 3D
segmentation model first — *not* the inherited heterogeneous ensemble.
The inherited 2.5D / cross-arch machinery is Phase 2 work after the
MONAI baseline produces credible Dice numbers.

## What is included vs excluded

Included:
- Source code (inherited Tier S verbatim + Tier A with DEFERRED_WMH_REVIEW stubs)
- nnU-Net trainer variant for MPS
- Cross-architecture ensemble script
- Audit map, citation metadata, roadmap, inheritance map
- Experiment-journey methodology doc

Not included:
- `Datasets/` — obtain WMH data separately
- Trained weights (`*.pt`, `*.pth`)
- `runs/`, `results/`, `logs/`
- ISLES configs and per-case evaluation artifacts (deliberately excluded)

## How to cite

See [CITATION.cff](CITATION.cff). The ISLES sibling repo is the upstream and
should also be cited if this repository's heterogeneous-ensemble or MPS 3D
workaround code is used in further work.

## License

Apache License 2.0 for code. See [LICENSE](LICENSE) and [NOTICE](NOTICE).


## 2026-06 MLOps refactor delta

This package now includes an executable local sequence for WMH2017 research PoC validation:

1. dataset manifest generation
2. label-value audit
3. train/validation split generation
4. one-case overlay visualization
5. MONAI 3D U-Net smoke training
6. validation prediction export
7. local metrics evaluation
8. run/evidence/release package manifest recording

Minimal command:

```bash
python -m pip install -r requirements-lock.txt
export WMH2017_ROOT=/path/to/MICCAI2017_WMH/files
bash scripts/run_wmh2017_minimal_pipeline.sh
```

Hard boundaries:
- Never use `challenge_split=test` for training, validation, threshold tuning, preprocessing fit, model selection, or early stopping.
- Never use `mask > 0` as foreground. WMH foreground is `label == 1`; `label == 2` is ignored as foreground.
- Do not use this for proprietary data, cloud execution, customer-facing reports, clinical decisions, or production.
- Do not make SOTA/leaderboard claims before official evaluation-code cross-check and review.

Current state:
- Structural checks: recorded separately; no release-state promotion
- Preview candidate only after required real-data evidence and human review
- Not `READY_FOR_RELEASE`.


### Official dataset download evidence

Non-raw download evidence for the Dataverse WMH2017 release is included under:

```text
evidence/wmh2017_download_2026-06-16/
```

Verify it with:

```bash
python scripts/verify_wmh2017_download_evidence.py \
  --evidence-dir evidence/wmh2017_download_2026-06-16 \
  --out reports/wmh2017_download_evidence_verification.json
```

Boundary: this confirms packaged acquisition/checksum evidence only. It is not
source/license approval, real training evidence, official benchmark parity,
clinical validation, customer approval, Preview, or Release.
