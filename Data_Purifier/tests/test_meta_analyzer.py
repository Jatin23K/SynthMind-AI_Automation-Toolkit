import os
import tempfile

import pandas as pd

from ..agents.meta_analyzer_agent import MetaAnalyzerAgent

# Sample test data
SAMPLE_DATA = {
    'order_id': [1001, 1002, 1003, 1004],
    'customer_name': ['John', 'Alice', 'Bob', 'Charlie'],
    'order_date': ['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18'],
    'total_amount': [150.50, 200.75, 99.99, 300.25],
    'product_category': ['Electronics', 'Clothing', 'Electronics', 'Home'],
    'payment_method': ['Credit', 'PayPal', 'Credit', 'Bank Transfer'],
    'is_premium': [True, False, False, True],
    'discount_percentage': [10, 0, 5, 15],
    'shipping_region': ['North', 'South', 'East', 'West'],
    'order_rating': [4.5, 3.5, 5.0, 4.0]
}

def create_test_files():
    """Create temporary test files and return their paths."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create test CSV file
    csv_path = os.path.join(temp_dir, 'test_orders.csv')
    pd.DataFrame(SAMPLE_DATA).to_csv(csv_path, index=False)

    # Create test meta file with questions and expected structure
    meta_path = os.path.join(temp_dir, 'test_meta.txt')
    with open(meta_path, 'w') as f:
        f.write("""
        schema:
          columns:
            - order_id
            - customer_name
            - order_date
            - total_amount
            - product_category
            - payment_method
            - is_premium
            - discount_percentage
            - shipping_region
            - order_rating
          rows: 4
          column_types:
            order_id: int
            customer_name: object
            order_date: datetime
            total_amount: float
            product_category: object
            payment_method: object
            is_premium: bool
            discount_percentage: int
            shipping_region: object
            order_rating: float
        questions:
          - What is the total sales by product category?
          - How many orders were placed by premium customers?
          - What is the average order amount by shipping region?
        pipeline_plan:
          - cleaning
          - modification
          - transformation
        suggested_operations:
          cleaning_operations:
            total_amount:
              - operation: handle_missing
                method: mean
                reason: Impute missing total amounts with the mean.
          modification_operations:
            total_amount:
              - operation: feature_engineering
                method: calculate_tax
                reason: Calculate tax based on total amount.
          transformation_operations:
            product_category:
              - operation: categorical_encoding
                method: one_hot
                reason: One-hot encode product categories.
        """)

    return csv_path, meta_path, temp_dir

def test_meta_analyzer_basic():
    """Test basic functionality of MetaAnalyzerAgent."""
    # Create test files
    csv_path, meta_path, temp_dir = create_test_files()

    try:
        # Initialize agent
        meta_analyzer = MetaAnalyzerAgent()

        # Run analysis
        df, instructions = meta_analyzer.analyze(num_datasets=1, dataset_paths=[csv_path], meta_output_path=meta_path)

        # Verify results
        assert not df.empty, "Resulting DataFrame should not be empty"
        assert 'pipeline_plan' in instructions, "Should return pipeline plan"
        assert 'suggested_operations' in instructions, "Should return suggested operations"

        # Check if relevant columns are in the analysis
        expected_columns = {'product_category', 'total_amount', 'is_premium', 'shipping_region'}
        # The relevant columns are now determined by the LLM based on the questions and schema
        # We can't directly assert specific columns unless we control the LLM output precisely.
        # Instead, we'll check if the structure is as expected.
        assert 'cleaning_operations' in instructions['suggested_operations'], "Should have cleaning operations"
        assert 'modification_operations' in instructions['suggested_operations'], "Should have modification operations"
        assert 'transformation_operations' in instructions['suggested_operations'], "Should have transformation operations"

        assert 'removed_columns' in instructions['analysis_report'], "Should have removed_columns in analysis_report"
        assert 'data_type_changes' in instructions['analysis_report'], "Should have data_type_changes in analysis_report"

        print("Basic Meta Analyzer Test: PASSED")

    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

def test_data_type_standardization():
    """Test if data types are properly standardized."""
    csv_path, meta_path, temp_dir = create_test_files()

    try:
        meta_analyzer = MetaAnalyzerAgent()
        df, meta_analysis_report = meta_analyzer.analyze(num_datasets=1, dataset_paths=[csv_path], meta_output_path=meta_path)

        # Check data types based on the meta_analysis_report's schema
        assert pd.api.types.is_numeric_dtype(df['total_amount']), "total_amount should be numeric"
        assert pd.api.types.is_bool_dtype(df['is_premium']), "is_premium should be boolean"

        print("Data Type Standardization Test: PASSED")

    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_empty_questions():
    """Test behavior when no questions are provided."""
    csv_path, _, temp_dir = create_test_files()

    try:
        # Create empty meta file
        empty_meta_path = os.path.join(temp_dir, 'empty_meta.txt')
        with open(empty_meta_path, 'w') as f:
            f.write("""No questions here""")

        meta_analyzer = MetaAnalyzerAgent()
        df, instructions = meta_analyzer.analyze(num_datasets=1, dataset_paths=[csv_path], meta_output_path=empty_meta_path)

        # Should keep all columns when no questions are provided
        assert len(df.columns) == len(SAMPLE_DATA), "Should keep all columns when no questions are provided"
        assert len(instructions['questions']) == 0, "Should have no questions"

        print("Empty Questions Test: PASSED")

    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_missing_data():
    """Test handling of missing data."""
    # Create test data with missing values
    test_data = SAMPLE_DATA.copy()
    test_data['total_amount'] = [150.50, None, 99.99, 300.25]  # Fixed: Don't modify original
    test_data['product_category'] = ['Electronics', 'Clothing', None, 'Home']  # Fixed: Don't modify original

    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, 'test_missing.csv')
    meta_path = os.path.join(temp_dir, 'test_meta.txt')

    try:
        # Save test data
        pd.DataFrame(test_data).to_csv(csv_path, index=False)

        # Create meta file
        with open(meta_path, 'w') as f:
            f.write("Question: What is the average order amount by product category?")

        meta_analyzer = MetaAnalyzerAgent()
        df, instructions = meta_analyzer.analyze(num_datasets=1, dataset_paths=[csv_path], meta_output_path=meta_path)

        # Check if data type information is captured
        assert 'column_types' in instructions['schema'], "Should include column type information"

        print("Missing Data Test: PASSED")

    finally:
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_meta_analyzer_basic()
    test_data_type_standardization()
    test_empty_questions()
    test_missing_data()
    print("All tests completed successfully!")
