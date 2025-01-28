import pytest
import pandas as pd
import asyncio
from rs2nda import ResponseMapper

@pytest.fixture
def cde_definitions():
    """Create a sample CDE definitions DataFrame."""
    data = {
        'ElementName': ['test_element'],
        'Notes': ['0 = Never; 1 = Sometimes; 2 = Often; 3 = Very Often; -9 = Missing'],
        'ElementDescription': ['Test description']
    }
    return pd.DataFrame(data)

@pytest.fixture
def response_mapper(cde_definitions):
    """Create a ResponseMapper instance."""
    return ResponseMapper(cde_definitions)

@pytest.mark.asyncio
async def test_exact_value_match(response_mapper):
    """Test when values match exactly."""
    response = {
        'response_value': 1,
        'response_options': [
            {'name': {'en': 'Never'}, 'value': 0},
            {'name': {'en': 'Sometimes'}, 'value': 1},
            {'name': {'en': 'Often'}, 'value': 2}
        ]
    }
    cde_mapping = {'Never': '0', 'Sometimes': '1', 'Often': '2'}
    result = await response_mapper._find_best_match(
        response['response_value'], 
        cde_mapping,
        response['response_options']
    )
    assert result == '1'

@pytest.mark.asyncio
async def test_semantic_match(response_mapper):
    """Test when meanings match but wording differs."""
    response = {
        'response_value': 1,
        'response_options': [
            {'name': {'en': 'Not at all'}, 'value': 0},
            {'name': {'en': 'A couple of times'}, 'value': 1},
            {'name': {'en': 'Regularly'}, 'value': 2}
        ]
    }
    cde_mapping = {'Never': '0', 'Sometimes': '1', 'Frequently': '2'}
    result = await response_mapper._find_best_match(
        response['response_value'], 
        cde_mapping,
        response['response_options']
    )
    # LLM should recognize 'A couple of times' is similar to 'Sometimes'
    assert result == '1'

@pytest.mark.asyncio
async def test_no_semantic_match(response_mapper):
    """Test when no semantic match exists."""
    response = {
        'response_value': 1,
        'response_options': [
            {'name': {'en': 'Completely different scale'}, 'value': 1}
        ]
    }
    cde_mapping = {'Never': '0', 'Sometimes': '1', 'Often': '2'}
    result = await response_mapper._find_best_match(
        response['response_value'], 
        cde_mapping,
        response['response_options']
    )
    assert result == '-9'

@pytest.mark.asyncio
async def test_complex_semantic_match(response_mapper):
    """Test complex semantic matching scenarios."""
    test_cases = [
        # Test case 1: Similar duration descriptions
        {
            'response': {
                'response_value': 1,
                'response_options': [{'name': {'en': 'Less than a day or two'}, 'value': 1}]
            },
            'cde_mapping': {'Rare, happens once or twice': '1'},
            'expected': '1'
        },
        # Test case 2: Similar frequency descriptions
        {
            'response': {
                'response_value': 2,
                'response_options': [{'name': {'en': 'Multiple times a week'}, 'value': 2}]
            },
            'cde_mapping': {'Several times per week': '2'},
            'expected': '2'
        },
        # Test case 3: Similar intensity descriptions
        {
            'response': {
                'response_value': 3,
                'response_options': [{'name': {'en': 'Very intensely'}, 'value': 3}]
            },
            'cde_mapping': {'Severe intensity': '3'},
            'expected': '3'
        }
    ]
    
    for case in test_cases:
        result = await response_mapper._find_best_match(
            case['response']['response_value'],
            case['cde_mapping'],
            case['response']['response_options']
        )
        assert result == case['expected'], f"Failed on case: {case}"

@pytest.mark.asyncio
async def test_full_mapping_workflow(response_mapper):
    """Test the entire mapping workflow with multiple responses."""
    reproschema_responses = [
        {
            'id': 'q1',
            'question': 'How often?',
            'response_value': 1,
            'response_options': [
                {'name': {'en': 'Once in a while'}, 'value': 1}
            ]
        },
        {
            'id': 'q2',
            'question': 'How severe?',
            'response_value': 2,
            'response_options': [
                {'name': {'en': 'Moderately bad'}, 'value': 2}
            ]
        }
    ]
    
    matched_mapping = {
        'element1': ('q1', 0.8),
        'element2': ('q2', 0.9)
    }
    
    mapped_data = await response_mapper.map_responses(
        reproschema_responses,
        matched_mapping
    )
    
    # Check that we got mappings for both elements
    assert 'element1' in mapped_data
    assert 'element2' in mapped_data
    
    # Check that the values are valid (either valid mappings or -9)
    for value in mapped_data.values():
        assert value in ['0', '1', '2', '3', '-9']

if __name__ == "__main__":
    pytest.main([__file__])