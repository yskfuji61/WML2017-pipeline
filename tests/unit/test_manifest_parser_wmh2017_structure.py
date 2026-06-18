from pathlib import Path

from wmh2017.data.manifest import build_manifest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")


def _make_case(root: Path, rel: str, with_wmh: bool = True) -> None:
    case = root / rel
    _touch(case / "pre" / "FLAIR.nii.gz")
    _touch(case / "pre" / "T1.nii.gz")
    _touch(case / "orig" / "FLAIR.nii.gz")
    _touch(case / "orig" / "T1.nii.gz")
    _touch(case / "orig" / "3DT1.nii.gz")
    if with_wmh:
        _touch(case / "wmh.nii.gz")


def test_manifest_parser_handles_wmh2017_layout(tmp_path):
    files = tmp_path / "files"
    _make_case(files, "training/Utrecht/0")
    _make_case(files, "training/Singapore/50")
    _make_case(files, "training/Amsterdam/GE3T/100")
    _make_case(files, "test/Amsterdam/GE1T5/150")
    _make_case(files, "test/Amsterdam/Philips_VU .PETMR_01./161")

    _touch(files / "additional_annotations" / "observer_o3" / "training" / "Utrecht" / "0" / "result.nii.gz")
    _touch(files / "additional_annotations" / "observer_o4" / "training" / "Amsterdam" / "GE3T" / "100" / "result.nii.gz")

    df = build_manifest(files)

    assert len(df) == 5
    assert set(df["challenge_split"]) == {"training", "test"}
    assert (df["flair_pre_path"].astype(str).str.endswith("pre/FLAIR.nii.gz")).all()
    assert (df["t1_pre_path"].astype(str).str.endswith("pre/T1.nii.gz")).all()

    utrecht = df[(df["site"] == "Utrecht") & (df["case_id"].astype(str) == "0")].iloc[0]
    assert utrecht["scanner_code"] == "utrecht_philips_achieva_3t"
    assert bool(utrecht["has_additional_o3"]) is True

    ams_ge3t = df[(df["site"] == "Amsterdam") & (df["case_id"].astype(str) == "100")].iloc[0]
    assert ams_ge3t["scanner_code"] == "amsterdam_ge_signa_hdxt_3t"
    assert bool(ams_ge3t["has_additional_o4"]) is True

    ams_ge1t5 = df[(df["site"] == "Amsterdam") & (df["case_id"].astype(str) == "150")].iloc[0]
    assert ams_ge1t5["challenge_split"] == "test"
    assert ams_ge1t5["scanner_code"] == "amsterdam_ge_signa_hdxt_1p5t"

    ams_philips = df[(df["site"] == "Amsterdam") & (df["case_id"].astype(str) == "161")].iloc[0]
    assert ams_philips["scanner_code"] == "amsterdam_philips_ingenuity_3t"


def test_manifest_accepts_parent_containing_files(tmp_path):
    root = tmp_path / "MICCAI2017_WMH"
    files = root / "files"
    _make_case(files, "training/Utrecht/0")

    df = build_manifest(root)

    assert len(df) == 1
    assert df.iloc[0]["case_id"] == "0"
