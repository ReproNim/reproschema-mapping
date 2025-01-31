import pytest
import pandas as pd
from rs2nda import QuestionMatcher
import asyncio

@pytest.fixture
def sample_cde_definitions():
    """Create sample CDE definitions for testing."""
    return pd.DataFrame({
        'ElementName': ['demo_age', 'demo_sex', 'demo_race', 'custom_element'],
        'ElementDescription': [
            'Age of the participant', 
            'Sex of the participant',
            'Race of the participant',
            'What is your favorite hobby?'  # Changed to test similarity matching
        ],
        'Aliases': [
            'interview_age,age',
            'sex,gender',
            'race,ethnicity,demo_white',
            None
        ]
    })

@pytest.fixture
def sample_reproschema_responses():
    """Create sample ReproSchema responses for testing."""
    return [
        {
            'id': 'demo_age',
            'question': 'What is your age?',
            'response_value': '25'
        },
        {
            'id': 'gender',
            'question': 'What is your gender?',
            'response_value': 'Female'
        },
        {
            'id': 'unmapped_id',
            'question': 'What are your hobbies and interests?', 
            'response_value': 'Reading'
        },
        {
            'id': 'another_id',
            'question': 'What is your favorite color?',
            'response_value': 'Blue'
        }
    ]

def test_exact_id_match():
    """Test direct matching between reproschema IDs and ElementNames."""
    matcher = QuestionMatcher()
    cde_definitions = pd.DataFrame({
        'ElementName': ['demo_age', 'demo_sex']
    })
    reproschema_ids = ['demo_age', 'other_id']
    
    matches, used_ids = matcher._exact_id_match(reproschema_ids, cde_definitions['ElementName'].tolist())
    
    assert 'demo_age' in matches
    assert matches['demo_age'] == ('demo_age', 1.0)
    assert 'demo_age' in used_ids
    assert 'demo_sex' not in matches

def test_alias_match(sample_cde_definitions):
    """Test matching through aliases."""
    matcher = QuestionMatcher()
    reproschema_ids = ['gender', 'unknown_id']
    used_ids = set()
    
    matches, newly_used_ids = matcher._alias_match(
        reproschema_ids, 
        sample_cde_definitions,
        used_ids
    )
    
    assert 'demo_sex' in matches
    assert matches['demo_sex'] == ('gender', 1.0)
    assert 'gender' in newly_used_ids

def test_similarity_match(sample_cde_definitions, sample_reproschema_responses):
    """Test semantic similarity matching."""
    matcher = QuestionMatcher()
    # Only mark exact matches as used, leaving 'unmapped_id' for similarity matching
    used_ids = {'demo_age', 'gender'}
    
    matches = matcher._similarity_match(
        sample_cde_definitions,
        sample_reproschema_responses,
        used_ids,
        threshold=0.5
    )
    
    # We expect 'unmapped_id' with question about hobbies 
    # to match with 'custom_element' due to semantic similarity
    assert len(matches) > 0
    # Check specific match
    custom_element_match = matches.get('custom_element')
    assert custom_element_match is not None
    assert custom_element_match[0] == 'unmapped_id'
    assert custom_element_match[1] >= 0.5  # Confidence should be above threshold

def test_similarity_match_threshold():
    """Test that similarity matching respects the threshold."""
    matcher = QuestionMatcher()
    
    cde_definitions = pd.DataFrame({
        'ElementName': ['test_element'],
        'ElementDescription': ['A very specific description about XYZ'],
        'Aliases': [None]
    })
    
    reproschema_responses = [
        {
            'id': 'test_id',
            'question': 'A completely unrelated question about ABC',
            'response_value': 'test'
        }
    ]
    
    # Should not match with high threshold
    high_threshold_matches = matcher._similarity_match(
        cde_definitions,
        reproschema_responses,
        set(),
        threshold=0.9
    )
    assert len(high_threshold_matches) == 0
    
    # Should match with low threshold
    low_threshold_matches = matcher._similarity_match(
        cde_definitions,
        reproschema_responses,
        set(),
        threshold=0.1
    )
    assert len(low_threshold_matches) == 1

@pytest.mark.asyncio
async def test_agent_match(sample_cde_definitions, sample_reproschema_responses):
    """Test agent-based matching."""
    matcher = QuestionMatcher()
    unmatched_responses = [
        r for r in sample_reproschema_responses 
        if r['id'] == 'complex_question'
    ]
    
    matches = await matcher._agent_match(unmatched_responses, sample_cde_definitions)
    
    assert isinstance(matches, dict)
    for element_name, (repro_id, confidence) in matches.items():
        assert isinstance(element_name, str)
        assert isinstance(repro_id, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

@pytest.mark.asyncio
async def test_full_matching_pipeline(sample_cde_definitions, sample_reproschema_responses):
    """Test the complete matching pipeline."""
    matcher = QuestionMatcher()
    
    matches = await matcher.match(
        sample_cde_definitions, 
        sample_reproschema_responses
    )
    
    # Check that we got matches
    assert len(matches) > 0
    
    # Check the structure of matches
    for element_name, (repro_id, confidence) in matches.items():
        assert element_name in sample_cde_definitions['ElementName'].values
        assert isinstance(repro_id, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        
    # Check specific expected matches
    assert 'demo_age' in matches
    assert matches['demo_age'][0] == 'demo_age'  # Exact match
    assert matches['demo_age'][1] == 1.0  # Perfect confidence

if __name__ == '__main__':
    pytest.main([__file__])