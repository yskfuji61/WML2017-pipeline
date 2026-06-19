"""Overclaim scanner unit tests with negative-context allowance."""

from __future__ import annotations

from wmh2017.security.overclaim import ClaimContext, classify_line, scan_text


def test_positive_ready_for_release_fails():
    hits = scan_text("This repository is READY_FOR_RELEASE.", source="sample.md")
    assert hits
    assert hits[0].context == ClaimContext.PROHIBITED_POSITIVE


def test_negative_ready_for_release_passes():
    hits = scan_text("READY_FOR_RELEASE must never be claimed.", source="sample.md")
    assert not hits


def test_japanese_clinical_overclaim_fails():
    hits = scan_text("本パッケージは臨床利用可能です。", source="sample.md")
    assert hits


def test_japanese_ai_diagnosis_overclaim_fails():
    hits = scan_text("AI診断できます。", source="sample.md")
    assert hits


def test_sota_negative_boundary_passes():
    assert classify_line("No SOTA claim is made.", "sota_positive_en") == ClaimContext.ALLOWED_NEGATIVE
    hits = scan_text("Future work must not include SOTA overclaim.", source="sample.md")
    assert not hits


def test_policy_table_reference_passes():
    hits = scan_text("| READY_FOR_RELEASE | Requires formal release decision |", source="crosswalk.md")
    assert not hits
