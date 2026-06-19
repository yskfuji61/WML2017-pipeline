#!/usr/bin/env python3
"""Fetch official WMH evaluator source under reviewed, pinned supply-chain controls."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_PIN = Path("registry/official_evaluator_pin.yaml")


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_pin(repo_root: Path, pin_path: Path) -> dict:
    data = yaml.safe_load(pin_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid pin file: {pin_path}")
    return data


def _validate_allowlist(repo_url: str, allowlist: list[str]) -> None:
    if repo_url not in allowlist:
        raise SystemExit(f"repo-url not in allowlist: {repo_url}")


def _write_source_record(
    path: Path,
    *,
    repo_url: str,
    commit: str,
    tree_sha256: str,
    license_review: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        "\n".join(
            [
                "# Official WMH evaluator source record",
                "",
                f"- URL: {repo_url}",
                "- Status: FETCHED",
                f"- Commit: {commit}",
                f"- Tree SHA256: {tree_sha256}",
                f"- License review: {license_review}",
                f"- Fetch date: {now}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official WMH evaluator (fail-closed).")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    parser.add_argument("--source-record", required=True)
    parser.add_argument("--license-review", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pin-file", default=str(DEFAULT_PIN))
    args = parser.parse_args()

    if args.license_review.upper() != "APPROVED":
        raise SystemExit(f"license review must be APPROVED, got {args.license_review}")

    repo_root = Path(__file__).resolve().parents[1]
    pin_path = repo_root / args.pin_file
    pin = _load_pin(repo_root, pin_path)
    allowlist = pin.get("allowlist_urls", [])
    if not isinstance(allowlist, list):
        raise SystemExit("pin allowlist_urls must be a list")
    _validate_allowlist(args.repo_url, [str(item) for item in allowlist])

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    source_record = Path(args.source_record)
    if not source_record.is_absolute():
        source_record = repo_root / source_record
    if not source_record.parent.exists():
        raise SystemExit(f"source record parent missing: {source_record.parent}")

    if output_dir.exists():
        existing_hash = _tree_sha256(output_dir)
        if existing_hash != args.expected_tree_sha256:
            raise SystemExit(
                f"existing output dir hash mismatch: expected {args.expected_tree_sha256}, got {existing_hash}"
            )
        print(f"output dir already present with matching hash: {output_dir}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", args.repo_url, str(output_dir)], cwd=str(repo_root))
    subprocess.check_call(["git", "checkout", args.commit], cwd=str(output_dir))

    actual_hash = _tree_sha256(output_dir)
    if actual_hash != args.expected_tree_sha256:
        raise SystemExit(f"tree sha256 mismatch after fetch: expected {args.expected_tree_sha256}, got {actual_hash}")

    sha_path = output_dir.parent / "evaluator.sha256"
    sha_path.write_text(actual_hash + "\n", encoding="utf-8")
    exact_commit_path = output_dir.parent / "exact_commit_or_archive.txt"
    exact_commit_path.write_text(args.commit + "\n", encoding="utf-8")
    _write_source_record(
        source_record,
        repo_url=args.repo_url,
        commit=args.commit,
        tree_sha256=actual_hash,
        license_review=args.license_review,
    )
    print(f"Fetched evaluator commit {args.commit} -> {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"fetch failed: {exc}") from exc
