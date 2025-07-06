import numpy as np
import pandas as pd

from ..agents.cleaner_agent import CleanerAgent
from ..agents.meta_analyzer_agent import MetaAnalyzerAgent
from ..agents.transformer_agent import TransformerAgent


def test_cleaner_transformer_integration():
    # Create test data with various issues
    test_df = pd.DataFrame({
        'numeric': [1, 2, np.nan, 2, 5, 100],  # Contains missing value and outlier
        'category': ['x', 'y', 'z', 'y', 'x', 'w'],  # Contains duplicates
        'to_normalize': [10, 20, 30, 20, 50, 60]
    })

    # Initialize agents
    cleaner = CleanerAgent()
    transformer = TransformerAgent()

    # Step 1: Clean the data
    cleaned_df, clean_report = cleaner.clean_dataset(
        df=test_df,
        cleaning_instructions={
            "remove_duplicates": [{'operation': 'remove_duplicates'}],
            "numeric": [{'operation': 'handle_missing_values', 'method': 'mean'},
                        {'operation': 'handle_outliers', 'method': 'isolation_forest'}],
            "category": [{'operation': 'handle_missing_values', 'method': 'mode'}]
        }
    )

    # Verify cleaning results
    assert all(op['status'] == 'completed' for op in clean_report['operations_performed_details']), "All cleaning operations should be completed"
    assert cleaned_df.isna().sum().sum() == 0, "No missing values should remain"
    assert len(cleaned_df) == len(cleaned_df.drop_duplicates()), "No duplicates should remain"
    assert 100 not in cleaned_df['numeric'].tolist(), "Outlier 100 should be handled"

    # Step 2: Transform the cleaned data
    transformed_df, transform_report = transformer.transform_data(
        df=cleaned_df,
        processing_instructions={
            'transformation_operations': {
                'numeric': [{'operation': 'scale_normalize', 'method': 'standard_scaler'}],
                'category': [{'operation': 'encode_categorical', 'method': 'one_hot'}]
            }
        }
    )

    assert all(op['status'] == 'completed' for op in transform_report['operations_performed']), "All transformation operations should be completed"
    assert 'category_x' in transformed_df.columns, "One-hot encoded column should be created"
    print("Integration Test (Cleaner -> Transformer): PASSED")

def test_end_to_end_pipeline():
    # Create test data
    test_df = pd.DataFrame({
        'numeric': [1, 2, np.nan, 2, 5, 100],
        'category': ['x', 'y', 'z', 'y', 'x', 'w'],
        'to_normalize': [10, 20, 30, 20, 50, 60]
    })

    # Initialize all agents
    cleaner = CleanerAgent()
    transformer = TransformerAgent()
    analyzer = MetaAnalyzerAgent()

    # Step 1: Analyze initial data
    import tempfile
    from pathlib import Path

    # Create dummy CSV and meta files
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_csv_path = Path(tmpdir) / "dummy.csv"
        dummy_meta_path = Path(tmpdir) / "dummy_meta.txt"

        pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']}).to_csv(dummy_csv_path, index=False)
        with open(dummy_meta_path, 'w') as f:
            f.write("""schema:
  columns:
    - col1
    - col2
  column_types:
    col1: int
    col2: object
questions:
  - What is the meaning of life?
pipeline_plan:
  - cleaning
suggested_operations:
  cleaning_operations:
    col1:
      - operation: handle_missing_values
        method: mean
        reason: Test operation
  transformation_operations:
    col2:
      - operation: encode_categorical
        method: one_hot
        reason: Test operation
""")

        initial_profile, _ = analyzer.analyze(num_datasets=1, dataset_paths=[str(dummy_csv_path)], meta_output_path=str(dummy_meta_path))
        assert initial_profile is not None, "Initial profiling should succeed"
        assert 'col1' in initial_profile.columns, "Expected column 'col1' not found in initial profile"

        # Step 2: Clean the data
        cleaned_df, clean_report = cleaner.clean_dataset(
            df=test_df,
            cleaning_instructions={
                "remove_duplicates": [{'operation': 'remove_duplicates'}],
                "numeric": [{'operation': 'handle_missing_values', 'method': 'mean'},
                            {'operation': 'handle_outliers', 'method': 'isolation_forest'}],
                "category": [{'operation': 'handle_missing_values', 'method': 'mode'}]
            }
        )
        assert all(op['status'] == 'completed' for op in clean_report['operations_performed_details']), "All cleaning operations should be completed"

        # Step 3: Transform the data
        transformed_df, transform_report = transformer.transform_data(
            df=cleaned_df,
            processing_instructions={
                'transformation_operations': {
                    'numeric': [{'operation': 'scale_normalize', 'method': 'standard_scaler'}],
                    'category': [{'operation': 'encode_categorical', 'method': 'one_hot'}]
                }
            }
        )
        assert all(op['status'] == 'completed' for op in transform_report['operations_performed']), "All transformation operations should be completed"

        print("End-to-end Pipeline Test: PASSED")