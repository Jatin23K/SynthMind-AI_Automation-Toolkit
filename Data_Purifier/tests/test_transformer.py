import pandas as pd

from ..agents.transformer_agent import TransformerAgent


def test_transformer():
    # Initialize agent
    transformer = TransformerAgent()

    # Create test data
    test_df = pd.DataFrame({
        'numeric_value': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'missing_data': [1.0, 2.0, None, 4.0, 5.0],
        'to_bin': [10, 20, 30, 40, 50]
    })

    # Test numeric scaling
    transformed_df, transformation_report = transformer.transform_data(
        df=test_df.copy(),
        processing_instructions={'transformation_operations': {'numeric_value': [{'operation': 'scale_normalize', 'method': 'standard_scaler'}]}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert all(op['status'] == 'completed' for op in transformation_report['operations_performed']), "All operations should succeed"
    assert abs(transformed_df['numeric_value'].mean()) < 1e-10, "Data should be standardized"
    assert abs(transformed_df['numeric_value'].std() - 1) < 1e-10, "Data should be standardized"

    # Test categorical encoding
    transformed_df, transformation_report = transformer.transform_data(
        df=test_df.copy(),
        processing_instructions={'transformation_operations': {'category': [{'operation': 'encode_categorical', 'method': 'one_hot'}]}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert all(op['status'] == 'completed' for op in transformation_report['operations_performed']), "All operations should succeed"
    assert 'category_A' in transformed_df.columns, "Encoded column should be created"
    assert 'category_B' in transformed_df.columns, "Encoded column should be created"
    assert 'category_C' in transformed_df.columns, "Encoded column should be created"

    # Test numeric binning
    transformed_df, transformation_report = transformer.transform_data(
        df=test_df.copy(),
        processing_instructions={'transformation_operations': {'to_bin': [{'operation': 'discretize_bin', 'method': 'equal_width', 'bins': 3}]}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert all(op['status'] == 'completed' for op in transformation_report['operations_performed']), "All operations should succeed"
    assert 'to_bin_binned' in transformed_df.columns, "Binned column should be created"
    assert len(transformed_df['to_bin_binned'].unique()) == 3, "Should have 3 bins"

    print("Transformer Tests: All PASSED")

def test_transformer_edge_cases():
    transformer = TransformerAgent()

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    transformed_empty_df, transformation_report = transformer.transform_data(
        df=empty_df,
        processing_instructions={'transformation_operations': {}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert not any(op['status'] == 'success' for op in transformation_report['operations_performed']), "No operations should succeed on empty DataFrame"
    assert transformed_empty_df.empty, "Transformed DataFrame should be empty"

    # Test with DataFrame with one row
    single_row_df = pd.DataFrame({'numeric_value': [1], 'category': ['A']})
    transformed_single_row_df, transformation_report = transformer.transform_data(
        df=single_row_df,
        processing_instructions={'transformation_operations': {'numeric_value': [{'operation': 'scale_normalize', 'method': 'standard_scaler'}], 'category': [{'operation': 'encode_categorical', 'method': 'one_hot'}]}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert all(op['status'] == 'completed' for op in transformation_report['operations_performed']), "All operations should succeed"
    assert len(transformed_single_row_df) == 1, "Transformed DataFrame should have one row"

    # Test scaling on non-numeric columns
    non_numeric_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
    transformed_non_numeric, transformation_report = transformer.transform_data(
        df=non_numeric_df.copy(),
        processing_instructions={'transformation_operations': {'A': [{'operation': 'scale_normalize', 'method': 'standard_scaler'}], 'B': [{'operation': 'scale_normalize', 'method': 'standard_scaler'}]}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert any(op['status'] == 'failed' for op in transformation_report['operations_performed']), "Scaling non-numeric should fail"
    assert 'A' in transformed_non_numeric.columns, "Non-numeric column should remain"
    assert abs(transformed_non_numeric['B'].mean()) < 1e-10, "Numeric column B should be standardized"

    # Test encoding on non-categorical columns
    non_categorical_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    transformed_non_categorical, transformation_report = transformer.transform_data(
        df=non_categorical_df.copy(),
        processing_instructions={'transformation_operations': {'A': [{'operation': 'encode_categorical', 'method': 'one_hot'}], 'B': [{'operation': 'encode_categorical', 'method': 'one_hot'}]}}
    )
    assert transformation_report, "Transformation report should not be empty"
    assert any(op['status'] == 'failed' for op in transformation_report['operations_performed']), "Encoding non-categorical should fail"
    assert 'A' in transformed_non_categorical.columns, "Non-categorical column should remain"
    assert 'B_x' in transformed_non_categorical.columns, "Categorical column B should be encoded"

    print("Transformer Edge Cases Test: PASSED")