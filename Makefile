PYTHON ?= python
RUN_ID ?= local_$(shell date +%Y%m%d_%H%M%S)
WMH2017_ROOT ?=
PACKAGE_ID ?= WMH2017-LOCAL-POC-SCAFFOLD
PACKAGE_VERSION ?= 0.2.3

.PHONY: setup lint typecheck test security sbom fingerprint manifest e2e verify-package verify-lineage verify-binder preview-candidate rollback-rehearsal parity-report

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e ".[dev,test,medical-image]"

lint:
	ruff check src scripts tests

typecheck:
	mypy src scripts

test:
	pytest -q tests/unit tests/integration tests/schema tests/contract tests/architecture tests/smoke tests/evaluation

security:
	mkdir -p reports/security
	detect-secrets scan --baseline .secrets.baseline --all-files > reports/security/detect_secrets.json
	detect-secrets audit .secrets.baseline --report > reports/security/detect_secrets_audit.txt
	bandit -q -r src scripts -f json -o reports/security/bandit.json || true
	pip-audit -r requirements-lock.txt -f json -o reports/security/pip_audit.json || true
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
	$(MAKE) verify-binder RUN_ID=$(RUN_ID)
	$(MAKE) verify-package RUN_ID=$(RUN_ID)
