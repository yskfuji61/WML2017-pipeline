def pytest_configure(config):
    config.addinivalue_line("markers", "requires_scipy: metric needs scipy.ndimage at runtime")
    config.addinivalue_line("markers", "requires_torch: needs torch/monai stack at runtime")
