from pathlib import Path


def test_overlay_png_gitignored():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    assert "reports/figures/overlays" in text


def test_visual_report_manifest_has_no_absolute_path():
    repo = Path(__file__).resolve().parents[1]
    import subprocess
    import sys

    out_dir = repo / "reports/figures"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/validation/generate_visual_report.py"),
            "--run-id",
            "test_visual",
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd=repo,
    )
    manifest = out_dir / "test_visual_figure_manifest.json"
    text = manifest.read_text(encoding="utf-8")
    assert "REDACTED_OR_LOCAL_ONLY" in text
    assert "/Users/" not in text
