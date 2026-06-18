import importlib.util


def test_monai_dependency_status_is_explicit():
    # CI can run without MONAI; real medical-image smoke is executed by
    # scripts/train_wmh2017.py when requirements-lock.txt medical stack is installed.
    assert importlib.util.find_spec("monai") is None or importlib.util.find_spec("torch") is not None
