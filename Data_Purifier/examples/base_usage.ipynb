# Data Purifier Basic Usage Example

This notebook demonstrates the basic usage of Data Purifier for cleaning and transforming your data.
## Setup
First, let's import the required libraries and create some sample data with common issues.
import pandas as pd
import numpy as np
from data_purifier.tasks.cleaner_tasks import DataCleaner
from data_purifier.tasks.transformer_tasks import DataTransformer

# Create sample data with various issues
np.random.seed(42)
n_samples = 1000

# Generate synthetic dataset
raw_data = {
    'age': np.random.normal(35, 10, n_samples),  # Some unrealistic ages
    'income': np.random.lognormal(10, 1, n_samples),  # Skewed distribution
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples)
}

# Add some data quality issues
df = pd.DataFrame(raw_data)
df.loc[np.random.choice(n_samples, 50), 'age'] = np.nan  # Add missing values
df.loc[np.random.choice(n_samples, 10), 'age'] = -5  # Add invalid ages
df.loc[np.random.choice(n_samples, 20), 'income'] = 0  # Add suspicious incomes

# Add some duplicate rows
df = pd.concat([df, df.iloc[np.random.choice(n_samples, 30)]])

print('Dataset shape:', df.shape)
df.head()
## Data Cleaning
Now let's clean the data using DataCleaner to handle missing values, outliers, and duplicates.
# Initialize the cleaner
cleaner = DataCleaner(df)

# Clean the data
cleaned_df = cleaner.remove_missing_values()\
                   .remove_duplicates()\
                   .handle_outliers()\
                   .get_result()

print('Cleaned dataset shape:', cleaned_df.shape)
cleaned_df.describe()
## Data Transformation
Next, let's transform the cleaned data by normalizing numeric columns and encoding categorical variables.
# Initialize the transformer
transformer = DataTransformer(cleaned_df)

# Transform the data
transformed_df = transformer.normalize_numeric_columns()\
                           .encode_categorical_variables()\
                           .get_result()

print('Transformed dataset shape:', transformed_df.shape)
transformed_df.head()
## Analysis of Results
Let's examine the effects of our data cleaning and transformation.
# Compare statistics before and after
print('Original Data Summary:
')
print(df.describe())
print('
Transformed Data Summary:
')
print(transformed_df.describe())

# Check for any remaining issues
print('
Missing values after cleaning:', transformed_df.isnull().sum())
print('Duplicate rows after cleaning:', transformed_df.duplicated().sum())
