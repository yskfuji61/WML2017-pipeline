"""Redacted medical image header audit (NIfTI-first)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NIFTI_FIELDS = ("descrip", "aux_file", "intent_name", "qform_present", "sform_present", "extensions_present")


@dataclass(frozen=True)
class HeaderAuditResult:
    path_redacted: str
    format: str
    status: str
    fields_checked: dict[str, str]
    raw_values_redacted: bool = True


def redact_path(path: Path) -> str:
    return "REDACTED_OR_LOCAL_ONLY"


def audit_nifti_header(path: Path) -> HeaderAuditResult:
    try:
        import nibabel as nib
    except ImportError:
        return HeaderAuditResult(
            path_redacted=redact_path(path),
            format="nifti",
            status="NOT_SUPPORTED_REQUIRES_OPTIONAL_DEPENDENCY",
            fields_checked={field: "NOT_SUPPORTED_REQUIRES_OPTIONAL_DEPENDENCY" for field in NIFTI_FIELDS},
        )

    img = nib.load(str(path))
    header = img.header
    fields_checked = {
        "descrip": "present" if getattr(header, "get_descrip", lambda: b"")() else "empty",
        "aux_file": "present" if getattr(header, "get_aux_file", lambda: b"")() else "empty",
        "intent_name": "present" if getattr(header, "get_intent_name", lambda: b"")() else "empty",
        "qform_present": "present" if header.get_qform(coded=False) is not None else "missing",
        "sform_present": "present" if header.get_sform(coded=False) is not None else "missing",
        "extensions_present": "present" if header.extensions else "empty",
    }
    return HeaderAuditResult(
        path_redacted=redact_path(path),
        format="nifti",
        status="AUDITED_REDACTED",
        fields_checked=fields_checked,
    )


def audit_file(path: Path) -> HeaderAuditResult:
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return audit_nifti_header(path)
    if lower.endswith(".dcm"):
        return HeaderAuditResult(
            path_redacted=redact_path(path),
            format="dicom",
            status="NOT_SUPPORTED_REQUIRES_OPTIONAL_DEPENDENCY",
            fields_checked={"dicom_tags": "NOT_SUPPORTED_REQUIRES_OPTIONAL_DEPENDENCY"},
        )
    if lower.endswith((".nrrd", ".mha", ".mhd")):
        return HeaderAuditResult(
            path_redacted=redact_path(path),
            format=path.suffix.lstrip("."),
            status="NOT_SUPPORTED_REQUIRES_OPTIONAL_DEPENDENCY",
            fields_checked={"header": "NOT_SUPPORTED_REQUIRES_OPTIONAL_DEPENDENCY"},
        )
    return HeaderAuditResult(
        path_redacted=redact_path(path),
        format="unknown",
        status="UNSUPPORTED_FORMAT",
        fields_checked={},
    )


def audit_root(root: Path, *, max_files: int = 50) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith((".nii", ".nii.gz", ".dcm", ".nrrd", ".mha", ".mhd")):
            files.append(path)
        if len(files) >= max_files:
            break

    results = [audit_file(path) for path in files]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": "REDACTED_OR_LOCAL_ONLY",
        "dlp_class": "PUBLIC_CHALLENGE_DATA",
        "raw_metadata_values_included": False,
        "files_audited": len(results),
        "results": [
            {
                "path": item.path_redacted,
                "format": item.format,
                "status": item.status,
                "fields_checked": item.fields_checked,
                "raw_values_redacted": item.raw_values_redacted,
            }
            for item in results
        ],
    }
    payload["audit_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return payload
