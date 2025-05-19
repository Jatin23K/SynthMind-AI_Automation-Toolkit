import pandas as pd
from data_purifier.agents.modifier_agent import DataModifierAgent

def test_modifier():
    # Initialize agent
    modifier = DataModifierAgent()
    
    # Create test data
    test_df = pd.DataFrame({
        'first_name': ['John', 'Jane'],
        'last_name': ['Doe', 'Smith'],
        'age': [30, 25],
        'city': ['New York', 'London'],
        'country': ['USA', 'UK']
    })
    
    # Test full name creation
    modified_df, success = modifier.process(
        df=test_df,
        processing_instructions={'create_full_name': True}
    )
    assert success, "Full name creation should succeed"
    assert 'full_name' in modified_df.columns, "Full name column should be created"
    assert modified_df['full_name'].iloc[0] == 'John Doe', "Full name should be correctly combined"
    
    # Test column combination
    modified_df, success = modifier.process(
        df=test_df,
        processing_instructions={
            'combine_columns': ['city', 'country'],
            'separator': ', '
        }
    )
    assert success, "Column combination should succeed"
    assert 'city_country' in modified_df.columns, "Combined column should be created"
    assert modified_df['city_country'].iloc[0] == 'New York, USA', "Columns should be correctly combined"
    
    # Test column renaming
    modified_df, success = modifier.process(
        df=test_df,
        processing_instructions={
            'rename_columns': {'age': 'years_old'}
        }
    )
    assert success, "Column renaming should succeed"
    assert 'years_old' in modified_df.columns, "Column should be renamed"
    assert 'age' not in modified_df.columns, "Old column name should not exist"
    
    # Test column dropping
    modified_df, success = modifier.process(
        df=test_df,
        processing_instructions={
            'drop_columns': ['city', 'country']
        }
    )
    assert success, "Column dropping should succeed"
    assert 'city' not in modified_df.columns, "Dropped column should not exist"
    assert 'country' not in modified_df.columns, "Dropped column should not exist"
    
    print("Modifier Tests: All PASSED")