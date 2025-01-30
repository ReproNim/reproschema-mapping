import pytest
import asyncio

# This tells pytest to treat all tests as async by default
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def event_loop_policy():
    """Return the event loop policy."""
    return asyncio.DefaultEventLoopPolicy()

# If we need to configure specific event loop parameters, we can use this mark
def pytest_configure(config):
    """Configure pytest with asyncio marks."""
    config.addinivalue_line(
        "markers", 
        "asyncio: mark test as requiring asyncio loop"
    )