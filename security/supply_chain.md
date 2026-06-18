# Supply chain

CI runs:

- `detect-secrets` → `reports/security/detect_secrets.json`
- `bandit` → `reports/security/bandit.json`
- `pip-audit` → `reports/security/pip_audit.json`

Install locked dependencies from `requirements-lock.txt` only for reproducible runs.
