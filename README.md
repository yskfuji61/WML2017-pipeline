# WML2017-pipeline

Auditable **offline** WMH2017 public-data research PoC package (not clinical, customer, cloud, or production software).

## Current state

| Field | Value |
|---|---|
| package_version | `0.2.3` |
| local planning state | `READY_FOR_STRUCTURAL_REVIEW` |
| controlled release state | `READY_FOR_PREVIEW` (structural package approved; not clinical/production) |
| first goal | `READY_FOR_PREVIEW` (not `READY_FOR_RELEASE`) |

**Blocked claims:** clinical use, customer presentation, proprietary-data processing, unapproved cloud upload, production deployment, SOTA / leaderboard equivalence.

Active package lives at repository root (`src/wmh2017/`).

## Quick start

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e ".[dev,test,medical-image]"
export WMH2017_ROOT=/path/to/MICCAI2017_WMH/files
make test
make e2e RUN_ID=wmh2017_preview_YYYYMMDD_gitsha WMH2017_ROOT="$WMH2017_ROOT"
make verify-package RUN_ID="$RUN_ID"
python scripts/verify_lineage_graph.py --run-id "$RUN_ID" --require-artifact-hashes
python scripts/verify_evidence_binder.py --run-id "$RUN_ID" --target-state READY_FOR_PREVIEW
```

Minimal smoke path:

```bash
bash scripts/run_wmh2017_minimal_pipeline.sh
```

Raw WMH2017 data stays **outside git**. Set `WMH2017_ROOT` in your shell or `.env.local` (gitignored).

## Audit entry points

- [README_CURSOR_START.md](README_CURSOR_START.md) — agent / developer handoff
- [docs/release_state_crosswalk.md](docs/release_state_crosswalk.md) — controlled release states
- [docs/release_evidence/README.md](docs/release_evidence/README.md) — preview promotion evidence index
- [docs/final_evidence_binder_index.md](docs/final_evidence_binder_index.md) — evidence binder
- [AUDIT_MAP.md](AUDIT_MAP.md) — repository map
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) — preview promotion checklist

## Active vs legacy paths

| Path | Role |
|---|---|
| `src/wmh2017/**` | **Active** WMH2017 MONAI smoke pipeline |
| `core/pipeline/**` | **Legacy** ISLES-derived reference (not primary smoke path) |

See [core/pipeline/LEGACY_README.md](core/pipeline/LEGACY_README.md).

## Language

[Full package README (EN)](README_PACKAGE.md) | [日本語](README_ja.md)
