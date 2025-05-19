import pandas as pd
import numpy as np
from data_purifier.agents.cleaner_agent_simple import DataCleanerAgent

def test_cleaner():
    # Initialize agent
    cleaner = DataCleanerAgent()
    
    # Create test data with known issues
    test_df = pd.DataFrame({
        'A': [1, 2, np.nan, 2, 5],
        'B': ['x', 'y', 'z', 'y', 'x']
    })
    
    # Test cleaning
    cleaned_df, success, summary = cleaner.process(
        df_input=test_df,
        processing_instructions={'remove_duplicates': True, 'handle_missing': True},
        recorder_agent=None
    )
    
    # Verify results
    assert success, "Cleaning should succeed"
    assert cleaned_df.isna().sum().sum() == 0, "No missing values should remain"
    assert len(cleaned_df) == len(cleaned_df.drop_duplicates()), "No duplicates should remain"
    
    print("Cleaner Test: PASSED")

def test_cleaner_edge_cases():
    cleaner = DataCleanerAgent()

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    cleaned_empty_df, success, summary = cleaner.process(
        df_input=empty_df,
        processing_instructions={'remove_duplicates': True, 'handle_missing': True},
        recorder_agent=None
    )
    assert success, "Cleaning empty DataFrame should succeed"
    assert cleaned_empty_df.empty, "Cleaned DataFrame should be empty"

    # Test with DataFrame with one row
    single_row_df = pd.DataFrame({'A': [1], 'B': ['x']})
    cleaned_single_row_df, success, summary = cleaner.process(
        df_input=single_row_df,
        processing_instructions={'remove_duplicates': True, 'handle_missing': True},
        recorder_agent=None
    )
    assert success, "Cleaning single row DataFrame should succeed"
    assert len(cleaned_single_row_df) == 1, "Cleaned DataFrame should have one row"

    # Test handle_missing with different strategies
    missing_df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    cleaned_mean, success_mean, _ = cleaner.process(
        df_input=missing_df.copy(),
        processing_instructions={'handle_missing': True, 'handle_missing_strategy': 'mean'},
        recorder_agent=None
    )
    assert success_mean, "Handling missing with mean should succeed"
    assert not cleaned_mean.isna().any().any(), "Mean filled DataFrame should have no missing values"
    assert cleaned_mean['A'].iloc[2] == missing_df['A'].mean(), "Mean filling for A should be correct"

    cleaned_median, success_median, _ = cleaner.process(
        df_input=missing_df.copy(),
        processing_instructions={'handle_missing': True, 'handle_missing_strategy': 'median'},
        recorder_agent=None
    )
    assert success_median, "Handling missing with median should succeed"
    assert not cleaned_median.isna().any().any(), "Median filled DataFrame should have no missing values"
    assert cleaned_median['A'].iloc[2] == missing_df['A'].median(), "Median filling for A should be correct"

    # Test handle_outliers with different methods
    outlier_df = pd.DataFrame({'A': [1, 2, 3, 4, 100, 5, 6]})
    cleaned_zscore, success_zscore, _ = cleaner.process(
        df_input=outlier_df.copy(),
        processing_instructions={'handle_outliers': True, 'handle_outliers_method': 'zscore', 'handle_outliers_threshold': 2},
        recorder_agent=None
    )
    assert success_zscore, "Handling outliers with zscore should succeed"
    assert 100 not in cleaned_zscore['A'].tolist(), "Outlier 100 should be handled by zscore"

    cleaned_iqr, success_iqr, _ = cleaner.process(
        df_input=outlier_df.copy(),
        processing_instructions={'handle_outliers': True, 'handle_outliers_method': 'iqr', 'handle_outliers_threshold': 1.5},
        recorder_agent=None
    )
    assert success_iqr, "Handling outliers with iqr should succeed"
    assert 100 not in cleaned_iqr['A'].tolist(), "Outlier 100 should be handled by iqr"

    print("Cleaner Edge Cases Test: PASSED")