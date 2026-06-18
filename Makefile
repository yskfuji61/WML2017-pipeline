PYTHON ?= python
RUN_ID ?= local_$(shell date +%Y%m%d_%H%M%S)
WMH2017_ROOT ?=

.PHONY: setup lint typecheck test security sbom fingerprint e2e verify-package verify-lineage verify-binder preview-candidate

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
	bandit -q -r src scripts -f json -o reports/security/bandit.json
	pip-audit -r requirements-lock.txt -f json -o reports/security/pip_audit.json
	$(PYTHON) scripts/enforce_security_policy.py reports/security

sbom:
	$(PYTHON) scripts/generate_sbom.py --out reports/security/sbom.spdx.json --license-out reports/security/license_report.json

fingerprint:
	$(PYTHON) scripts/print_runtime_fingerprint.py \
	  --out artifacts/runs/$(RUN_ID)/runtime_fingerprint.json

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
	  --package-id WMH2017-LOCAL-POC-SCAFFOLD-0.0.0.0 \
	  --package-version 0.0.0.0

verify-lineage:
	$(PYTHON) scripts/verify_lineage_graph.py --run-id $(RUN_ID)

verify-binder:
	$(PYTHON) scripts/verify_evidence_binder.py \
	  --run-id $(RUN_ID) \
	  --target-state READY_FOR_PREVIEW

preview-candidate: setup lint typecheck test security sbom fingerprint e2e verify-package verify-lineage verify-binder
