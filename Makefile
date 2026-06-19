ifeq ($(wildcard .venv/bin/python),)
PYTHON ?= python3
else
PYTHON ?= .venv/bin/python
endif
RUN_ID ?= local_$(shell date +%Y%m%d_%H%M%S)
WMH2017_ROOT ?=
EPOCHS ?=
FULL_CONFIG ?= configs/wmh2017_monai_unet3d_full.yaml
PACKAGE_ID ?= WMH2017-LOCAL-POC-SCAFFOLD
PACKAGE_VERSION ?= 0.2.3

.PHONY: setup doctor lint typecheck test security sbom fingerprint manifest sync-manifests e2e e2e-full verify-package verify-lineage verify-binder preview-candidate rollback-rehearsal parity-report

doctor:
	$(PYTHON) scripts/check_environment.py

sync-manifests:
	$(PYTHON) scripts/data/sync_v4_manifests_from_csv.py \
	  --dataset-csv reports/dataset_manifest.csv \
	  --split-csv data/splits/wmh2017_train_val_seed42.csv

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e ".[dev,test,medical-image]"

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests

typecheck:
	$(PYTHON) -m mypy src scripts

test:
	$(PYTHON) -m pytest -q tests/unit tests/integration tests/schema tests/contract tests/architecture tests/smoke tests/evaluation tests/security tests/supply_chain

security:
	mkdir -p reports/security
	detect-secrets scan --baseline .secrets.baseline --all-files > reports/security/detect_secrets.json || true
	test -s reports/security/detect_secrets.json || echo '{}' > reports/security/detect_secrets.json
	detect-secrets audit .secrets.baseline --report > reports/security/detect_secrets_audit.txt
	touch reports/security/.detect_secrets.completed
	bandit -q -r src scripts -f json -o reports/security/bandit.json || test -s reports/security/bandit.json
	touch reports/security/.bandit.completed
	pip-audit -r requirements-lock.txt -f json -o reports/security/pip_audit.json || test -s reports/security/pip_audit.json
	touch reports/security/.pip_audit.completed
	$(PYTHON) scripts/generate_sbom.py --cdx-out reports/security/sbom.cdx.json --license-out reports/security/license_report.json
	$(PYTHON) scripts/enforce_security_policy.py reports/security

sbom:
	$(PYTHON) scripts/generate_sbom.py --cdx-out reports/security/sbom.cdx.json --license-out reports/security/license_report.json

manifest:
	mkdir -p reports
	$(PYTHON) scripts/verify_release_package.py \
	  --repo-root . \
	  --structural-only \
	  --out reports/full_package_manifest.json \
	  --package-id $(PACKAGE_ID) \
	  --package-version $(PACKAGE_VERSION)

fingerprint:
	$(PYTHON) scripts/print_runtime_fingerprint.py \
	  --out artifacts/runs/$(RUN_ID)/runtime_fingerprint.json

rollback-rehearsal:
	$(PYTHON) scripts/run_rollback_rehearsal.py --all-scenarios
	$(PYTHON) scripts/verify_rollback_rehearsal.py --target-state READY_FOR_PREVIEW

parity-report:
	$(PYTHON) scripts/generate_official_parity_report.py

e2e:
	test -n "$(WMH2017_ROOT)"
	$(PYTHON) scripts/run_wmh2017_e2e.py \
	  --files-root "$(WMH2017_ROOT)" \
	  --work-dir "artifacts/runs/$(RUN_ID)" \
	  --run-id "$(RUN_ID)"

e2e-full:
	test -n "$(WMH2017_ROOT)"
	$(PYTHON) scripts/run_wmh2017_e2e.py \
	  --files-root "$(WMH2017_ROOT)" \
	  --work-dir "artifacts/runs/$(RUN_ID)" \
	  --run-id "$(RUN_ID)" \
	  --config "$(FULL_CONFIG)" \
	  $(if $(EPOCHS),--max-epochs $(EPOCHS),)

verify-package:
	$(PYTHON) scripts/verify_release_package.py \
	  --repo-root . \
	  --run-dir artifacts/runs/$(RUN_ID) \
	  --out artifacts/runs/$(RUN_ID)/release/release_package_manifest.json \
	  --package-id $(PACKAGE_ID) \
	  --package-version $(PACKAGE_VERSION)

verify-lineage:
	$(PYTHON) scripts/verify_lineage_graph.py \
	  --run-id $(RUN_ID) \
	  --require-artifact-hashes \
	  --require-source-review \
	  --require-release-decision

verify-binder:
	$(PYTHON) scripts/verify_evidence_binder.py \
	  --run-id $(RUN_ID) \
	  --target-state READY_FOR_PREVIEW

preview-candidate: setup lint typecheck test parity-report rollback-rehearsal
	test -n "$(WMH2017_ROOT)"
	rm -rf artifacts/runs/$(RUN_ID)
	$(MAKE) e2e RUN_ID=$(RUN_ID) WMH2017_ROOT="$(WMH2017_ROOT)"
	$(MAKE) security
	$(MAKE) manifest
	$(PYTHON) scripts/sync_release_manifest_hashes.py --run-id $(RUN_ID)
	$(PYTHON) scripts/verify_package_identity.py
	$(PYTHON) scripts/verify_finding_register.py --target-state READY_FOR_PREVIEW
	$(PYTHON) scripts/validate_metric_table.py artifacts/runs/$(RUN_ID)/evaluation/case_metrics.csv
	$(MAKE) verify-lineage RUN_ID=$(RUN_ID)
	$(PYTHON) scripts/verify_release_evidence_register.py --run-id $(RUN_ID)
	$(MAKE) verify-binder RUN_ID=$(RUN_ID)
	$(MAKE) verify-package RUN_ID=$(RUN_ID)
