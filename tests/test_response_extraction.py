import pytest
from rs2nda import extract_reproschema_responses, get_constraints_url
import json
from unittest.mock import AsyncMock, MagicMock
from rs2nda import URLCache

@pytest.fixture
def sample_response_data():
    return [
        {
            "@type": "reproschema:Response",
            "@id": "response_1",
            "value": "test_value",
            "isAbout": "https://example.com/items/test_item"
        }
    ]

@pytest.fixture
def sample_item_content():
    return {
        "id": "test_item",
        "question": {
            "en": "Test question?"
        },
        "responseOptions": "../valueConstraints"
    }

@pytest.fixture
def sample_constraints_content():
    return {
        "choices": [
            {
                "name": {"en": "Option 1"},
                "value": 0
            },
            {
                "name": {"en": "Option 2"},
                "value": 1
            }
        ]
    }

@pytest.fixture
def mock_url_cache(monkeypatch):
    cache = AsyncMock()
    cache.fetch_item_content.side_effect = lambda url: {
        'test_url': {
            'id': 'test_id',
            'question': {'en': 'test question'},
            'responseOptions': None
        }
    }.get(url, {})
    monkeypatch.setattr('rs2nda.rs2nda.url_cache', cache)
    return cache

@pytest.mark.asyncio
async def test_response_extraction_with_valid_data(mock_url_cache):
    input_data = [{
        '@type': 'reproschema:Response',
        'isAbout': 'test_url',
        'value': 'test_value'
    }]
    
    result = await extract_reproschema_responses(input_data)
    assert len(result) == 1
    assert result[0]['id'] == 'test_id'

@pytest.mark.asyncio
async def test_response_extraction_with_missing_data(mock_url_cache):
    """Test response extraction with missing or invalid data."""
    response_data = [
        {
            "@type": "reproschema:Response",
            "value": "test_value"
            # Missing isAbout URL
        }
    ]
    
    result = await extract_reproschema_responses(response_data)
    assert len(result) == 1
    assert result[0]['id'] == ''  # Should have empty ID for missing data

def test_constraints_url_construction():
    """Test construction of constraints URLs."""
    test_cases = [
        {
            'is_about_url': 'https://example.com/items/test',
            'response_options': '../valueConstraints',
            'expected': 'https://example.com/valueConstraints'
        },
        {
            'is_about_url': 'https://example.com/nested/items/test',
            'response_options': '../constraints',
            'expected': 'https://example.com/nested/constraints'
        },
        {
            'is_about_url': 'invalid_url',
            'response_options': '../constraints',
            'expected': None
        }
    ]
    
    for case in test_cases:
        result = get_constraints_url(case['is_about_url'], case['response_options'])
        assert result == case['expected']

@pytest.mark.asyncio
async def test_response_extraction_error_handling(mock_url_cache):
    """Test error handling during response extraction."""
    # Configure mock to raise exception
    mock_url_cache.fetch_item_content.side_effect = Exception("Test error")
    
    response_data = [
        {
            "@type": "reproschema:Response",
            "value": "test_value",
            "isAbout": "https://example.com/items/test_item"
        }
    ]
    
    result = await extract_reproschema_responses(response_data)
    assert len(result) == 1
    assert result[0].get('response_options') is None