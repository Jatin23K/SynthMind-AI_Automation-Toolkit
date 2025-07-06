import numpy as np
import pandas as pd

from ..agents.cleaner_agent import CleanerAgent


def test_cleaner():
    # Initialize agent
    cleaner = CleanerAgent()

    # Create test data with known issues
    test_df = pd.DataFrame({
        'A': [1, 2, np.nan, 2, 5],
        'B': ['x', 'y', 'z', 'y', 'x']
    })

    # Test cleaning
    cleaned_df, cleaning_stats = cleaner.clean_dataset(
        df=test_df,
        cleaning_instructions={
            "remove_duplicates": [{'operation': 'remove_duplicates'}],
            "A": [{'operation': 'handle_missing_values', 'method': 'mean'}],
            "B": [{'operation': 'handle_missing_values', 'method': 'mode'}]
        }
    )

    # Verify results
    assert cleaned_df.isna().sum().sum() == 0, "No missing values should remain"
    assert len(cleaned_df) == len(cleaned_df.drop_duplicates()), "No duplicates should remain"
    assert cleaning_stats['operations_performed_details'][0]['status'] == 'completed', "Duplicate removal should be completed"
    assert cleaning_stats['operations_performed_details'][1]['status'] == 'completed', "Missing value handling for A should be completed"
    assert cleaning_stats['operations_performed_details'][2]['status'] == 'completed', "Missing value handling for B should be completed"

    print("Cleaner Test: PASSED")

def test_cleaner_edge_cases():
    cleaner = CleanerAgent()

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    cleaned_empty_df, cleaning_stats = cleaner.clean_dataset(
        df=empty_df,
        cleaning_instructions={}
    )
    assert cleaned_empty_df.empty, "Cleaned DataFrame should be empty"

    # Test with DataFrame with one row
    single_row_df = pd.DataFrame({'A': [1], 'B': ['x']})
    cleaned_single_row_df, cleaning_stats = cleaner.clean_dataset(
        df=single_row_df,
        cleaning_instructions={}
    )
    assert len(cleaned_single_row_df) == 1, "Cleaned DataFrame should have one row"

    # Test handle_missing with different strategies
    missing_df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    cleaned_mean, cleaning_stats_mean = cleaner.clean_dataset(
        df=missing_df.copy(),
        cleaning_instructions={
            "A": [{'operation': 'handle_missing_values', 'method': 'mean'}],
            "B": [{'operation': 'handle_missing_values', 'method': 'mean'}]
        }
    )
    assert not cleaned_mean.isna().any().any(), "Mean filled DataFrame should have no missing values"
    assert np.isclose(cleaned_mean['A'].mean(), missing_df['A'].mean()), "Mean filling for A should be correct"

    cleaned_median, cleaning_stats_median = cleaner.clean_dataset(
        df=missing_df.copy(),
        cleaning_instructions={
            "A": [{'operation': 'handle_missing_values', 'method': 'median'}],
            "B": [{'operation': 'handle_missing_values', 'method': 'median'}]
        }
    )
    assert not cleaned_median.isna().any().any(), "Median filled DataFrame should have no missing values"
    assert np.isclose(cleaned_median['A'].median(), missing_df['A'].median()), "Median filling for A should be correct"

    # Test handle_outliers with isolation_forest (as it's implemented)
    outlier_df = pd.DataFrame({'A': [1, 2, 3, 4, 100, 5, 6]})
    cleaned_outliers, cleaning_stats_outliers = cleaner.clean_dataset(
        df=outlier_df.copy(),
        cleaning_instructions={'A': [{'operation': 'handle_outliers', 'method': 'isolation_forest'}]}
    )
    assert 100 not in cleaned_outliers['A'].tolist(), "Outlier 100 should be handled by isolation_forest"

    print("Cleaner Edge Cases Test: PASSED")