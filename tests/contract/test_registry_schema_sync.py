import csv
from pathlib import Path

from wmh2017.evaluation.metric_schema import REQUIRED_CASE_METRIC_COLUMNS
from wmh2017.registry.schema_validation import csv_header_from_schema, load_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "registry"


def test_split_manifest_csv_header_matches_schema():
    schema_columns = set(csv_header_from_schema("split_manifest.schema.json"))
    csv_header = (
        REGISTRY_DIR.joinpath("split_manifest_schema.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    )
    assert set(csv_header) == schema_columns


def test_run_evidence_schema_required_keys_in_csv_register():
    schema_required = set(load_schema("run_evidence.schema.json").get("required", []))
    with REGISTRY_DIR.joinpath("run_evidence_schema_wmh2017.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_required = {row["field"] for row in rows if row["required"].lower() == "true"}
    assert "run_id" in schema_required
    assert "run_id" in csv_required
    undocumented = sorted(schema_required - csv_required - {"status", "safety"})
    assert not undocumented, f"run_evidence schema keys not in CSV register: {undocumented}"


def test_metric_result_schema_required_columns_in_metric_schema():
    schema_required = set(load_schema("metric_result.schema.json").get("required", []))
    missing = sorted(schema_required - set(REQUIRED_CASE_METRIC_COLUMNS))
    assert not missing, f"metric_result schema keys missing from metric_schema: {missing}"
