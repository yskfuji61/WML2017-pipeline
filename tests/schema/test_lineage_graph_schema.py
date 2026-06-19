from wmh2017.registry.schema_validation import validate_required_keys

FIXTURE = {
    "lineage_id": "lineage-test-001",
    "run_id": "wmh2017_preview_test",
    "package_version": "0.2.3",
    "code_commit": "abc123",
    "nodes": [{"id": "dataset", "type": "dataset"}],
    "edges": [{"from": "dataset", "to": "split"}],
}


def test_lineage_graph_fixture_passes_required_key_validation():
    failures = validate_required_keys(FIXTURE, "lineage_graph.schema.json")
    assert failures == []


def test_lineage_graph_missing_run_id_is_reported():
    payload = dict(FIXTURE)
    del payload["run_id"]
    failures = validate_required_keys(payload, "lineage_graph.schema.json")
    assert any("run_id" in failure for failure in failures)
