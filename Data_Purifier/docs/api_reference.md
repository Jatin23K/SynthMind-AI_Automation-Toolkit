# Data Purifier API Reference

This document provides detailed information about the Data Purifier API, including classes, methods, and their parameters.

## Core Components

### DataCleaner

The `DataCleaner` class provides methods for cleaning and preprocessing data.

```python
from data_purifier.tasks.cleaner_tasks import DataCleaner

cleaner = DataCleaner(df)
```

#### Methods

- `remove_missing_values(strategy='drop')`: Handles missing values in the dataset
  - Parameters:
    - `strategy`: String, optional (default='drop')
      - 'drop': Remove rows with missing values
      - 'mean': Fill missing values with mean
      - 'median': Fill missing values with median
      - 'mode': Fill missing values with mode

- `remove_duplicates(subset=None)`: Removes duplicate rows from the dataset
  - Parameters:
    - `subset`: List of columns to consider for duplicates, optional

- `handle_outliers(method='zscore', threshold=3)`: Detects and handles outliers
  - Parameters:
    - `method`: String, optional (default='zscore')
      - 'zscore': Use Z-score method
      - 'iqr': Use Interquartile Range method
    - `threshold`: Float, optional (default=3)
      - Z-score or IQR multiplier for outlier detection

- `get_result()`: Returns the cleaned DataFrame

### DataTransformer

The `DataTransformer` class provides methods for data transformation and feature engineering.

```python
from data_purifier.tasks.transformer_tasks import DataTransformer

transformer = DataTransformer(df)
```

#### Methods

- `normalize_numeric_columns(method='zscore')`: Normalizes numeric columns
  - Parameters:
    - `method`: String, optional (default='zscore')
      - 'zscore': Standardization (zero mean, unit variance)
      - 'minmax': Min-Max scaling to [0,1] range

- `encode_categorical_variables(method='label')`: Encodes categorical variables
  - Parameters:
    - `method`: String, optional (default='label')
      - 'label': Label encoding
      - 'onehot': One-hot encoding

- `get_result()`: Returns the transformed DataFrame

### MetaAnalyzer

The `MetaAnalyzer` class provides methods for data profiling and analysis.

```python
from data_purifier.tasks.meta_analyzer_tasks import MetaAnalyzer

analyzer = MetaAnalyzer(df)
```

#### Methods

- `generate_profile()`: Generates a comprehensive data profile report
- `get_summary_statistics()`: Returns basic summary statistics
- `get_correlation_matrix()`: Returns correlation matrix for numeric columns

## Configuration

### Environment Variables

Data Purifier uses the following environment variables:

- `OPENAI_API_KEY`: Required for advanced text analysis features
- `LOG_LEVEL`: Optional, sets the logging level (default: INFO)

Create a `.env` file in your project root:

```plaintext
OPENAI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## Error Handling

Data Purifier provides detailed error messages and exceptions:

- `DataQualityError`: Raised when data quality issues are detected
- `TransformationError`: Raised when data transformation fails
- `ConfigurationError`: Raised when configuration is invalid

## Quick Start

To get started with Data Purifier, follow these steps:

1. **Installation:**
   Install the library using pip:
   ```bash
   pip install data-purifier
   ```

2. **Basic Usage:**
   Load your data and apply basic cleaning and transformation:
   ```python
   import pandas as pd
   from data_purifier.tasks.cleaner_tasks import DataCleaner
   from data_purifier.tasks.transformer_tasks import DataTransformer

   # Load your data (replace 'your_data.csv' with your file path)
   df = pd.read_csv('your_data.csv')

   # Clean the data
   cleaner = DataCleaner(df)
   cleaned_df = cleaner.remove_missing_values(strategy='drop').remove_duplicates().get_result()

   # Transform the data
   transformer = DataTransformer(cleaned_df)
   transformed_df = transformer.normalize_numeric_columns(method='zscore').get_result()

   print("Original DataFrame head:")
   print(df.head())
   print("\nCleaned and Transformed DataFrame head:")
   print(transformed_df.head())
   ```

## Best Practices

Here are some best practices for using Data Purifier effectively:

1. **Start with Data Profiling:**
   Always begin by understanding your dataset's characteristics, including missing values, outliers, and data types, using the `MetaAnalyzer`.
   ```python
   import pandas as pd
   from data_purifier.tasks.meta_analyzer_tasks import MetaAnalyzer

   df = pd.read_csv('your_data.csv')
   analyzer = MetaAnalyzer(df)
   profile_report = analyzer.generate_profile()
   print("Data Profile Report:\n", profile_report)

   summary_stats = analyzer.get_summary_statistics()
   print("\nSummary Statistics:\n", summary_stats)

   correlation_matrix = analyzer.get_correlation_matrix()
   print("\nCorrelation Matrix:\n", correlation_matrix)
   ```

2. **Clean Data Incrementally:**
   Apply cleaning steps like handling missing values and duplicates before proceeding to transformations. Chain methods for a concise workflow.
   ```python
   import pandas as pd
   from data_purifier.tasks.cleaner_tasks import DataCleaner

   df = pd.read_csv('your_data.csv')
   cleaner = DataCleaner(df)
   cleaned_df = cleaner.remove_missing_values(strategy='mean').remove_duplicates().get_result()
   print("Cleaned DataFrame head:\n", cleaned_df.head())
   ```

3. **Choose Appropriate Transformations:**
   Select transformation methods based on your data and the requirements of your downstream tasks (e.g., machine learning models). Use normalization for numerical data and encoding for categorical data.
   ```python
   import pandas as pd
   from data_purifier.tasks.transformer_tasks import DataTransformer

   # Assuming 'cleaned_df' is available from the previous step
   transformer = DataTransformer(cleaned_df)
   transformed_df = transformer\
       .normalize_numeric_columns(method='minmax')\
       .encode_categorical_variables(method='onehot')\
       .get_result()
   print("Transformed DataFrame head:\n", transformed_df.head())
   ```

4. **Monitor and Validate:**
   After applying cleaning and transformation steps, always validate the results and monitor the process, especially in production pipelines. Use logging to track the execution.
   ```python
   import logging
   import pandas as pd
   from data_purifier.tasks.cleaner_tasks import DataCleaner

   logging.basicConfig(level=logging.INFO)

   df = pd.read_csv('your_data.csv')
   logging.info("Starting data cleaning...")
   cleaner = DataCleaner(df)
   cleaned_df = cleaner.remove_missing_values().get_result()
   logging.info("Data cleaning finished.")
   print("Cleaned DataFrame head:\n", cleaned_df.head())
   ```

5. **Handle Errors Gracefully:**
   Implement error handling using the provided exceptions (`DataQualityError`, `TransformationError`, `ConfigurationError`) to manage issues during data processing.
   ```python
   import pandas as pd
   from data_purifier.tasks.cleaner_tasks import DataCleaner
   from data_purifier.core.errors import DataQualityError

   df = pd.DataFrame({'col1': [1, 2, None, 4]})
   cleaner = DataCleaner(df)
   try:
       cleaned_df = cleaner.remove_missing_values(strategy='invalid').get_result()
   except DataQualityError as e:
       print(f"Caught a data quality error: {e}")
   ```
