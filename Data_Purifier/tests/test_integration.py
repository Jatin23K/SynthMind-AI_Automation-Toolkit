import pandas as pd
import numpy as np
from data_purifier.agents.cleaner_agent_simple import DataCleanerAgent
from data_purifier.agents.transformer_agent import DataTransformerAgent
from data_purifier.agents.meta_analyzer_agent import MetaAnalyzerAgent

def test_cleaner_transformer_integration():
    # Create test data with various issues
    test_df = pd.DataFrame({
        'numeric': [1, 2, np.nan, 2, 5, 100],  # Contains missing value and outlier
        'category': ['x', 'y', 'z', 'y', 'x', 'w'],  # Contains duplicates
        'to_normalize': [10, 20, 30, 20, 50, 60]
    })
    
    # Initialize agents
    cleaner = DataCleanerAgent()
    transformer = DataTransformerAgent()
    
    # Step 1: Clean the data
    cleaned_df, clean_success, clean_summary = cleaner.process(
        df_input=test_df,
        processing_instructions={
            'remove_duplicates': True,
            'handle_missing': True,
            'handle_outliers': True
        },
        recorder_agent=None
    )
    
    # Verify cleaning results
    assert clean_success, "Cleaning should succeed"
    assert cleaned_df.isna().sum().sum() == 0, "No missing values should remain"
    assert len(cleaned_df) == len(cleaned_df.drop_duplicates()), "No duplicates should remain"
    
    # Step 2: Transform the cleaned data
    transformed_df, transform_success = transformer.process(
        df=cleaned_df,
        processing_instructions={
            'scale_numeric': True,
            'encode_categorical': True
        }
    )
    
    # Verify transformation results
    assert transform_success, "Transformation should succeed"
    assert 'category_encoded' in transformed_df.columns, "Categories should be encoded"
    assert abs(transformed_df['numeric'].mean()) < 1e-10, "Numeric data should be standardized"
    
    print("Integration Test (Cleaner -> Transformer): PASSED")

def test_end_to_end_pipeline():
    # Create test data
    test_df = pd.DataFrame({
        'numeric': [1, 2, np.nan, 2, 5, 100],
        'category': ['x', 'y', 'z', 'y', 'x', 'w'],
        'to_normalize': [10, 20, 30, 20, 50, 60]
    })
    
    # Initialize all agents
    cleaner = DataCleanerAgent()
    transformer = DataTransformerAgent()
    analyzer = MetaAnalyzerAgent()
    
    # Step 1: Analyze initial data
    initial_profile = analyzer.process(test_df)
    assert initial_profile is not None, "Initial profiling should succeed"
    
    # Step 2: Clean the data
    cleaned_df, clean_success, _ = cleaner.process(
        df_input=test_df,
        processing_instructions={
            'remove_duplicates': True,
            'handle_missing': True,
            'handle_outliers': True
        },
        recorder_agent=None
    )
    assert clean_success, "Cleaning should succeed"
    
    # Step 3: Transform the data
    transformed_df, transform_success = transformer.process(
        df=cleaned_df,
        processing_instructions={
            'scale_numeric': True,
            'encode_categorical': True
        }
    )
    assert transform_success, "Transformation should succeed"
    
    # Step 4: Analyze final data
    final_profile = analyzer.process(transformed_df)
    assert final_profile is not None, "Final profiling should succeed"
    
    print("End-to-end Pipeline Test: PASSED")