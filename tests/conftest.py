def pytest_configure(config):
    config.addinivalue_line("markers", "requires_scipy: metric needs scipy.ndimage at runtime")
