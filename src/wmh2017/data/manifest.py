"""WMH2017 manifest generation for the Dataverse 2022 release layout.

Expected root is the Dataverse `files` directory:

    $WMH2017_ROOT/
      training/
      test/
      additional_annotations/

The scanner is conservative:
- It records paths and metadata, but never copies raw NIfTI files into the repo.
- It exposes both `challenge_split` and legacy-compatible `source_split`.
- It treats `test` as heldout, even if test labels exist in the 2022 public release.
- It records additional observer annotations only as auxiliary evidence, not as the
  primary training reference.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wmh2017.io.images import load_image_metadata


NIFTI_SUFFIXES = (".nii", ".nii.gz")
TOP_LEVEL_SPLITS = {"training", "test"}

SCANNER_BY_SITE_OR_CODE = {
    ("Utrecht", ""): "3T Philips Achieva",
    ("Singapore", ""): "3T Siemens TrioTim",
    ("Amsterdam", "GE3T"): "3T GE Signa HDxt",
    ("Amsterdam", "GE1T5"): "1.5T GE Signa HDxt",
    ("Amsterdam", "Philips_VU .PETMR_01."): "3T Philips Ingenuity",
}

# Stable short scanner codes for CSV-friendly values.
SCANNER_CODE_BY_RAW = {
    "3T Philips Achieva": "utrecht_philips_achieva_3t",
    "3T Siemens TrioTim": "singapore_siemens_triotim_3t",
    "3T GE Signa HDxt": "amsterdam_ge_signa_hdxt_3t",
    "1.5T GE Signa HDxt": "amsterdam_ge_signa_hdxt_1p5t",
    "3T Philips Ingenuity": "amsterdam_philips_ingenuity_3t",
}

PRIMARY_RELATIVE_PATH_COLUMNS = {
    "flair_pre": "flair_pre_path",
    "t1_pre": "t1_pre_path",
    "wmh": "wmh_path",
}


@dataclass(frozen=True)
class CaseIdentity:
    challenge_split: str
    site: str
    scanner_dir: str
    scanner: str
    scanner_code: str
    case_id: str
    case_dir: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def load_sha256sums(path: str | Path | None) -> dict[str, str]:
    """Load a standard `sha256sum` file into a relative-path -> hash map.

    Expected line format:
        <hex_sha256><two spaces><relative/path>
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SHA256SUMS file not found: {p}")

    mapping: dict[str, str] = {}
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid SHA256SUMS line {line_no}: {line!r}")
        digest, rel = parts
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"invalid SHA256 digest at line {line_no}: {digest}")
        mapping[rel.strip()] = digest.lower()
    return mapping


def _path_if_exists(path: Path) -> str:
    return str(path) if path.exists() else ""


def _hash_if_requested(path: Path, hash_files: bool) -> str:
    if not hash_files or not path.exists() or not path.is_file():
        return ""
    return sha256_file(path)


def _find_files_root(root: Path) -> Path:
    """Accept either the `files` root or a parent containing `files`."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    if (root / "training").is_dir() or (root / "test").is_dir():
        return root

    if (root / "files").is_dir():
        candidate = root / "files"
        if (candidate / "training").is_dir() or (candidate / "test").is_dir():
            return candidate

    # Backward-compatible fallback: root may be the training or test directory.
    if root.name in TOP_LEVEL_SPLITS:
        return root.parent

    return root


def _relative_to_files_root(path: Path, files_root: Path) -> str:
    try:
        return str(path.relative_to(files_root.parent))
    except ValueError:
        try:
            return str(path.relative_to(files_root))
        except ValueError:
            return str(path)


def _expected_sha256(path: Path, files_root: Path, sha256sums: dict[str, str]) -> str:
    if not sha256sums or not path.exists():
        return ""
    candidates = [
        str(path),
        str(path.relative_to(files_root)) if files_root in path.parents or path == files_root else "",
        _relative_to_files_root(path, files_root),
    ]
    for candidate in candidates:
        if candidate and candidate in sha256sums:
            return sha256sums[candidate]
    return ""


def infer_case_identity(case_dir: Path, files_root: Path) -> CaseIdentity:
    rel = case_dir.relative_to(files_root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"not a WMH2017 case directory: {case_dir}")

    challenge_split = parts[0]
    if challenge_split not in TOP_LEVEL_SPLITS:
        raise ValueError(f"unknown challenge split in path: {case_dir}")

    site = parts[1]
    if site == "Amsterdam":
        if len(parts) < 4:
            raise ValueError(f"Amsterdam case path must include scanner directory: {case_dir}")
        scanner_dir = parts[2]
        case_id = parts[3]
    else:
        scanner_dir = ""
        case_id = parts[2]

    scanner = SCANNER_BY_SITE_OR_CODE.get((site, scanner_dir), scanner_dir or "unknown")
    scanner_code = SCANNER_CODE_BY_RAW.get(scanner, (scanner_dir or site).lower().replace(" ", "_"))
    return CaseIdentity(
        challenge_split=challenge_split,
        site=site,
        scanner_dir=scanner_dir,
        scanner=scanner,
        scanner_code=scanner_code,
        case_id=str(case_id),
        case_dir=case_dir,
    )


def iter_case_dirs(files_root: Path) -> list[Path]:
    files_root = _find_files_root(files_root)
    case_dirs: list[Path] = []

    for split in ("training", "test"):
        split_dir = files_root / split
        if not split_dir.is_dir():
            continue

        for candidate in sorted(split_dir.rglob("*")):
            if not candidate.is_dir():
                continue
            # A WMH case directory has at least pre/ or orig/ folders.
            if (candidate / "pre").is_dir() or (candidate / "orig").is_dir():
                try:
                    infer_case_identity(candidate, files_root)
                except ValueError:
                    continue
                case_dirs.append(candidate)

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for case_dir in case_dirs:
        key = str(case_dir)
        if key not in seen:
            seen.add(key)
            out.append(case_dir)
    return out


def _additional_annotation_map(files_root: Path, observer: str) -> dict[tuple[str, str, str], str]:
    """Map (site, scanner_code, case_id) -> observer result path."""
    files_root = _find_files_root(files_root)
    root = files_root / "additional_annotations" / observer / "training"
    mapping: dict[tuple[str, str, str], str] = {}
    if not root.is_dir():
        return mapping

    for result_path in sorted(root.rglob("result.nii.gz")):
        # Reconstruct the equivalent training case directory under files_root.
        rel = result_path.parent.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            continue
        if parts[0] == "Amsterdam":
            if len(parts) < 3:
                continue
            case_dir = files_root / "training" / parts[0] / parts[1] / parts[2]
        else:
            case_dir = files_root / "training" / parts[0] / parts[1]

        try:
            ident = infer_case_identity(case_dir, files_root)
        except ValueError:
            continue
        mapping[(ident.site, ident.scanner_code, ident.case_id)] = str(result_path)

    return mapping


def _metadata_record(path: Path, prefix: str, inspect_images: bool) -> dict[str, str]:
    if not inspect_images or not path.exists():
        return {
            f"{prefix}_shape": "",
            f"{prefix}_spacing": "",
            f"{prefix}_dtype": "",
            f"{prefix}_affine_sha256": "",
            f"{prefix}_metadata_status": "not_inspected" if path.exists() else "missing",
        }
    try:
        meta = load_image_metadata(path)
    except Exception as exc:
        return {
            f"{prefix}_shape": "",
            f"{prefix}_spacing": "",
            f"{prefix}_dtype": "",
            f"{prefix}_affine_sha256": "",
            f"{prefix}_metadata_status": f"error:{type(exc).__name__}:{exc}",
        }
    return {
        f"{prefix}_shape": "x".join(str(x) for x in meta.shape),
        f"{prefix}_spacing": "x".join(f"{float(x):.8g}" for x in meta.spacing),
        f"{prefix}_dtype": meta.dtype,
        f"{prefix}_affine_sha256": meta.affine_sha256,
        f"{prefix}_metadata_status": "ok",
    }


def build_manifest(
    root: Path,
    hash_files: bool = False,
    *,
    inspect_images: bool = False,
    sha256sums: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build one row per WMH2017 case.

    Parameters
    ----------
    root:
        Prefer the Dataverse `files` directory, e.g.
        `/.../MICCAI2017_WMH/files`.
    hash_files:
        If true, compute SHA256 for primary pre/FLAIR, pre/T1, and wmh files.
        This is slower and should be used only for evidence snapshots.
    inspect_images:
        If true, inspect NIfTI geometry metadata for primary image/label files.
        This requires nibabel but does not copy raw image data into the repo.
    sha256sums:
        Optional expected checksums from the download evidence package. These
        are recorded without hashing raw files unless `hash_files=True`.
    """
    files_root = _find_files_root(root)
    expected = sha256sums or {}
    o3_map = _additional_annotation_map(files_root, "observer_o3")
    o4_map = _additional_annotation_map(files_root, "observer_o4")

    rows: list[dict] = []
    for case_dir in iter_case_dirs(files_root):
        ident = infer_case_identity(case_dir, files_root)
        key = (ident.site, ident.scanner_code, ident.case_id)

        flair_pre = case_dir / "pre" / "FLAIR.nii.gz"
        t1_pre = case_dir / "pre" / "T1.nii.gz"
        flair_orig = case_dir / "orig" / "FLAIR.nii.gz"
        t1_orig = case_dir / "orig" / "T1.nii.gz"
        t1_3d_orig = case_dir / "orig" / "3DT1.nii.gz"
        wmh = case_dir / "wmh.nii.gz"
        o3 = o3_map.get(key, "")
        o4 = o4_map.get(key, "")

        row = {
            "dataset_id": "WMH2017",
            "case_id": ident.case_id,
            "challenge_split": ident.challenge_split,
            # Backward-compatible alias used by split utilities.
            "source_split": ident.challenge_split,
            "site": ident.site,
            "scanner": ident.scanner,
            "scanner_code": ident.scanner_code,
            "scanner_dir": ident.scanner_dir,
            "case_dir": str(case_dir),
            "case_rel_dir": _relative_to_files_root(case_dir, files_root),
            "flair_pre_path": _path_if_exists(flair_pre),
            "t1_pre_path": _path_if_exists(t1_pre),
            "flair_orig_path": _path_if_exists(flair_orig),
            "t1_orig_path": _path_if_exists(t1_orig),
            "t1_3d_orig_path": _path_if_exists(t1_3d_orig),
            "wmh_path": _path_if_exists(wmh),
            "has_wmh": wmh.exists(),
            "has_additional_o3": bool(o3),
            "has_additional_o4": bool(o4),
            "additional_o3_path": o3,
            "additional_o4_path": o4,
            # Backward-compatible columns used by existing scripts.
            "flair_path": _path_if_exists(flair_pre) or _path_if_exists(flair_orig),
            "t1_path": _path_if_exists(t1_pre) or _path_if_exists(t1_orig),
            "mask_path": _path_if_exists(wmh),
            "flair_sha256": _hash_if_requested(flair_pre, hash_files),
            "t1_sha256": _hash_if_requested(t1_pre, hash_files),
            "mask_sha256": _hash_if_requested(wmh, hash_files),
            "flair_expected_sha256": _expected_sha256(flair_pre, files_root, expected),
            "t1_expected_sha256": _expected_sha256(t1_pre, files_root, expected),
            "mask_expected_sha256": _expected_sha256(wmh, files_root, expected),
            "shape": "",
            "spacing": "",
            "affine_hash": "",
            "label_values": "",
            "has_label2": "",
            "notes": "",
        }
        row.update(_metadata_record(flair_pre if flair_pre.exists() else flair_orig, "flair", inspect_images))
        row.update(_metadata_record(t1_pre if t1_pre.exists() else t1_orig, "t1", inspect_images))
        row.update(_metadata_record(wmh, "wmh", inspect_images))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["challenge_split", "site", "scanner_code", "case_id"]).reset_index(drop=True)
        df["created_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        duplicates = df[df["case_id"].astype(str).duplicated(keep=False)]
        if not duplicates.empty:
            cols = [
                c
                for c in ["challenge_split", "site", "scanner_code", "case_id"]
                if c in duplicates.columns
            ]
            raise ValueError(
                "case_id is not globally unique; evaluation currently joins by case_id only. "
                f"Duplicate rows: {duplicates[cols].to_dict(orient='records')}"
            )
    return df
