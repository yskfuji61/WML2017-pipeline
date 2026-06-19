#!/usr/bin/env python3
"""MLflow local config test."""

from pathlib import Path
import subprocess
import sys


def test_local_mlflow_config():
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(repo / "scripts/mlflow/setup_mlflow_experiment.py"), "--config", "configs/mlflow/local_mlflow.yaml"],
        check=True,
        cwd=repo,
    )
