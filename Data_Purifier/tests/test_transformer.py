import pandas as pd
from data_purifier.agents.transformer_agent import DataTransformerAgent

def test_transformer():
    # Initialize agent
    transformer = DataTransformerAgent()
    
    # Create test data
    test_df = pd.DataFrame({
        'numeric_value': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'missing_data': [1.0, 2.0, None, 4.0, 5.0],
        'to_bin': [10, 20, 30, 40, 50]
    })
    
    # Test numeric scaling
    transformed_df, success = transformer.process(
        df=test_df.copy(),
        processing_instructions={'scale_numeric': True}
    )
    assert success, "Numeric scaling should succeed"
    assert abs(transformed_df['numeric_value'].mean()) < 1e-10, "Data should be standardized"
    assert abs(transformed_df['numeric_value'].std() - 1) < 1e-10, "Data should be standardized"
    
    # Test categorical encoding
    transformed_df, success = transformer.process(
        df=test_df.copy(),
        processing_instructions={'encode_categorical': True}
    )
    assert success, "Categorical encoding should succeed"
    assert 'category_encoded' in transformed_df.columns, "Encoded column should be created"
    assert len(transformed_df['category_encoded'].unique()) == 3, "Should have 3 unique encoded values"
    
    # Test missing value filling
    transformed_df, success = transformer.process(
        df=test_df.copy(),
        processing_instructions={
            'fill_missing': True,
            'fill_strategy': 'mean'
        }
    )
    assert success, "Missing value filling should succeed"
    assert not transformed_df['missing_data'].isnull().any(), "No missing values should remain"
    assert transformed_df['missing_data'].mean() == 3.0, "Mean filling should be correct"
    
    # Test numeric binning
    transformed_df, success = transformer.process(
        df=test_df.copy(),
        processing_instructions={
            'bin_numeric': ['to_bin'],
            'n_bins': 3
        }
    )
    assert success, "Numeric binning should succeed"
    assert 'to_bin_binned' in transformed_df.columns, "Binned column should be created"
    assert len(transformed_df['to_bin_binned'].unique()) == 3, "Should have 3 bins"
    
    print("Transformer Tests: All PASSED")

def test_transformer_edge_cases():
    transformer = DataTransformerAgent()

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    transformed_empty_df, success = transformer.process(
        df=empty_df,
        processing_instructions={'scale_numeric': True, 'encode_categorical': True}
    )
    assert success, "Transforming empty DataFrame should succeed"
    assert transformed_empty_df.empty, "Transformed DataFrame should be empty"

    # Test with DataFrame with one row
    single_row_df = pd.DataFrame({'numeric_value': [1], 'category': ['A']})
    transformed_single_row_df, success = transformer.process(
        df=single_row_df,
        processing_instructions={'scale_numeric': True, 'encode_categorical': True}
    )
    assert success, "Transforming single row DataFrame should succeed"
    assert len(transformed_single_row_df) == 1, "Transformed DataFrame should have one row"

    # Test scaling on non-numeric columns
    non_numeric_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
    transformed_non_numeric, success = transformer.process(
        df=non_numeric_df.copy(),
        processing_instructions={'scale_numeric': True}
    )
    assert success, "Scaling with non-numeric columns should succeed"
    assert 'A' in transformed_non_numeric.columns, "Non-numeric column should remain"
    assert abs(transformed_non_numeric['B'].mean()) < 1e-10, "Numeric column B should be standardized"

    # Test encoding on non-categorical columns
    non_categorical_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    transformed_non_categorical, success = transformer.process(
        df=non_categorical_df.copy(),
        processing_instructions={'encode_categorical': True}
    )
    assert success, "Encoding with non-categorical columns should succeed"
    assert 'A' in transformed_non_categorical.columns, "Non-categorical column should remain"
    assert 'B_encoded' in transformed_non_categorical.columns, "Categorical column B should be encoded"

    print("Transformer Edge Cases Test: PASSED")