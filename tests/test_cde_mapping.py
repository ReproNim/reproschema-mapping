import pytest
import pandas as pd
from rs2nda import CDEMapping, ResponseMapper

@pytest.fixture
def sample_cde_mapping():
    return CDEMapping(
        numeric_to_string={'0': 'No', '1': 'Yes', '-9': 'Missing'},
        string_to_numeric={'No': '0', 'Yes': '1', 'Missing': '-9'},
        valid_values={'0', '1', '-9'},
        value_range='0::1'
    )

def test_cde_mapping_valid_values(sample_cde_mapping):
    """Test validation of values in CDEMapping."""
    assert sample_cde_mapping.is_valid_value('0')
    assert sample_cde_mapping.is_valid_value('1')
    assert sample_cde_mapping.is_valid_value('-9')
    assert not sample_cde_mapping.is_valid_value('2')
    assert not sample_cde_mapping.is_valid_value('invalid')

def test_cde_mapping_numeric_range():
    """Test numeric range validation in CDEMapping."""
    mapping = CDEMapping(
        min_value=0,
        max_value=5,
        valid_values=set(str(i) for i in range(6))
    )
    
    assert mapping.is_valid_value('0')
    assert mapping.is_valid_value('5')
    assert not mapping.is_valid_value('6')
    assert not mapping.is_valid_value('-1')

def test_parse_cde_notes():
    """Test parsing of CDE notes into mapping."""
    response_mapper = ResponseMapper(pd.DataFrame({
        'ElementName': ['test'],
        'DataType': ['String'],
        'ElementDescription': ['test'],
        'Notes': '0=No;1=Yes;2=Maybe;-9=Missing',
        'ValueRange': '0::2;-9'
    }))
    
    mapping = response_mapper._parse_cde_notes(
        '0=No;1=Yes;2=Maybe;-9=Missing',
        '0::2;-9'
    )
    
    assert mapping.numeric_to_string == {'0': 'No', '1': 'Yes', '2': 'Maybe', '-9': 'Missing'}
    assert mapping.string_to_numeric == {'No': '0', 'Yes': '1', 'Maybe': '2', 'Missing': '-9'}
    assert mapping.valid_values == {'0', '1', '2', '-9'}
    assert mapping.min_value == 0
    assert mapping.max_value == 2

def test_value_conversion():
    """Test conversion of values based on CDE mapping."""
    response_mapper = ResponseMapper(pd.DataFrame({
        'ElementName': ['test'],
        'DataType': ['Integer'],
        'ElementDescription': ['test'],
        'Notes': '0=No;1=Yes;-9=Missing',
        'ValueRange': '0::1;-9'
    }))
    
    # Test integer conversion
    assert response_mapper._validate_and_convert_value('1', 'Integer', 
                                                     response_mapper._get_cde_mapping('test')) == '1'
    assert response_mapper._validate_and_convert_value('invalid', 'Integer', 
                                                     response_mapper._get_cde_mapping('test')) == '-9'
    
    # Test string conversion
    assert response_mapper._validate_and_convert_value('Yes', 'String', 
                                                     response_mapper._get_cde_mapping('test')) == '1'
    assert response_mapper._validate_and_convert_value('invalid', 'String', 
                                                     response_mapper._get_cde_mapping('test')) == 'NR'

def test_sex_value_conversion():
    """Test conversion of sex values."""
    response_mapper = ResponseMapper(pd.DataFrame({
        'ElementName': ['sex'],
        'DataType': ['String'],
        'ElementDescription': ['Sex'],
        'Notes': 'M=Male;F=Female;NR=Not Reported',
        'ValueRange': None
    }))
    
    # Test various sex value formats
    assert response_mapper._convert_sex_value('http://schema.org/Female') == 'F'
    assert response_mapper._convert_sex_value('Male') == 'M'
    assert response_mapper._convert_sex_value('F') == 'F'
    assert response_mapper._convert_sex_value('invalid') == 'NR'