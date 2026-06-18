PYTHON ?= python
RUN_ID ?= local_$(shell date +%Y%m%d_%H%M%S)
WMH2017_ROOT ?=

.PHONY: setup lint typecheck test security fingerprint e2e verify-package preview-candidate

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e ".[dev,test,medical-image]"

lint:
	ruff check src scripts tests

typecheck:
	mypy src scripts

test:
	pytest -q tests/unit tests/integration tests/schema tests/contract tests/architecture tests/smoke

security:
	mkdir -p reports/security
	detect-secrets scan --all-files --force-use-all-plugins > reports/security/detect_secrets.json || true
	bandit -q -r src scripts -f json -o reports/security/bandit.json || true
	pip-audit -r requirements-lock.txt -f json -o reports/security/pip_audit.json || true

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
	  --out artifacts/runs/$(RUN_ID)/release/release_package_manifest.json

preview-candidate: setup lint typecheck test security fingerprint e2e verify-package
