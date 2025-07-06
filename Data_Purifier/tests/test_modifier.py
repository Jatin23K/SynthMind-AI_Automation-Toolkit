import pandas as pd

from ..agents.modifier_agent import DataModifierAgent


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
    modified_df, modification_report = modifier.modify_dataset(
        df=test_df,
        processing_instructions={
            'modification_operations': {
                'full_name': [{'operation': 'create_new_feature', 'method': 'combine_text', 'formula': "df['first_name'] + ' ' + df['last_name']", 'new_feature_name': 'full_name'}]
            }
        }
    )
    assert modification_report, "Modification report should not be empty"
    assert all(op['status'] == 'completed' for op in modification_report['operations_performed']), "All operations should succeed"
    assert 'full_name' in modified_df.columns, "Full name column should be created"
    assert modified_df['full_name'].iloc[0] == 'John Doe', "Full name should be correctly combined"

    # Test column renaming
    modified_df_rename, modification_report_rename = modifier.modify_dataset(
        df=test_df.copy(),
        processing_instructions={
            'modification_operations': {
                'age': [{'operation': 'rename_column', 'old_name': 'age', 'new_name': 'person_age'}]
            }
        }
    )
    assert 'person_age' in modified_df_rename.columns, "Column should be renamed"
    assert 'age' not in modified_df_rename.columns, "Original column should be dropped"

    # Test column dropping
    modified_df_drop, modification_report_drop = modifier.modify_dataset(
        df=test_df.copy(),
        processing_instructions={
            'modification_operations': {
                'city': [{'operation': 'drop_column', 'column_name': 'city'}]
            }
        }
    )
    assert 'city' not in modified_df_drop.columns, "Column should be dropped"

    print("Modifier Tests: All PASSED")