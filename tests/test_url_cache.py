import pytest
from unittest.mock import AsyncMock
from rs2nda import URLCache
import json

class MockContextManager:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockResponse:
    def __init__(self, data):
        self._data = data

    async def text(self):
        return json.dumps(self._data)

class MockSession:
    def __init__(self):
        self.closed = False
        self._default_response = {"data": "test"}

    def get(self, url):
        response = MockResponse(self._default_response)
        return MockContextManager(response)

    async def close(self):
        self.closed = True

@pytest.fixture
async def url_cache():
    """Create a URLCache instance with mocked session."""
    cache = URLCache()
    cache._session = MockSession()
    return cache

@pytest.mark.asyncio
async def test_basic_fetch(url_cache):
    """Test basic fetch functionality"""
    result = await url_cache.fetch_item_content("https://example.com/test")
    assert result == {"data": "test"}

@pytest.mark.asyncio
async def test_cache_hit(url_cache):
    """Test that cache returns cached values"""
    # Add something to cache directly
    url_cache._cache["test_url"] = {"cached": "data"}
    result = await url_cache.fetch_item_content("test_url")
    assert result == {"cached": "data"}

@pytest.mark.asyncio
async def test_cache_size_limit(url_cache):
    """Test cache size limit"""
    # Set smaller cache size
    url_cache.max_size = 2
    
    # Add three items
    await url_cache.fetch_item_content("url1")
    await url_cache.fetch_item_content("url2")
    await url_cache.fetch_item_content("url3")
    
    assert len(url_cache._cache) == 2
    assert "url1" not in url_cache._cache