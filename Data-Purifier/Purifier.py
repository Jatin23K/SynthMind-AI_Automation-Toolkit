import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from crewai import Agent, Task, Crew, Process
import logging
from sklearn.ensemble import IsolationForest
from pandas_profiling import ProfileReport
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import CountVectorizer

# Load API Key for OpenAI
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key

# Placeholder for input mechanism and further implementation
# ... existing code ...

def get_input_files():
    num_datasets = int(input("Enter the number of datasets: "))
    dataset_paths = [input(f"Enter the file path for dataset {i+1}: ") for i in range(num_datasets)]
    meta_output_path = os.path.join(os.path.dirname(__file__), "meta_output.txt")
    return dataset_paths, meta_output_path

# Define AI Agents
cleaner_agent = Agent(
    role="Data Cleaner",
    goal="Automatically clean datasets based on metadata",
    backstory="Expert in missing values, outliers, date handling, type correction, and format standardization.",
    verbose=True
)

modifier_agent = Agent(
    role="Data Modifier",
    goal="Modify and engineer dataset features",
    backstory="Expert in feature engineering, column standardization, renaming, encoding, and mapping",
    verbose=True
)

transformer_agent = Agent(
    role="Data Transformer",
    goal="Perform transformations like normalization, joins, binning, etc.",
    backstory="Expert in mathematical transformations, pivoting, and data structure manipulation.",
    verbose=True
)

summarizer_agent = Agent(
    role="Data Summarizer",
    goal="Summarize the entire cleaning, modification, and transformation journey.",
    backstory="Expert in reporting changes with full process breakdown.",
    verbose=True
)

# Define Tasks
cleaning_task = Task(
    description="Clean the dataset using metadata and intelligent process reasoning. Output each process in detailed explainable format.",
    agent=cleaner_agent,
    expected_output="Cleaned DataFrame with summary logs."
)

modifying_task = Task(
    description="Modify the dataset columns, labels, and categorical types using metadata.",
    agent=modifier_agent,
    expected_output="Modified DataFrame with feature improvements."
)

transforming_task = Task(
    description="Apply transformations such as normalization, binning, and log operations where needed.",
    agent=transformer_agent,
    expected_output="Transformed dataset ready for modeling."
)

summary_task = Task(
    description="Summarize the full pipeline of cleaning, modifying, and transforming in human-readable format.",
    agent=summarizer_agent,
    expected_output="Summary report of all steps performed on the data."
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Automated Feature Selection

def select_features(df, target_column):
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        logging.info(f"Selected features: {selected_features}")
        return df[selected_features]
    return df

# Anomaly Detection

def detect_anomalies(df):
    model = IsolationForest(contamination=0.1)
    df['anomaly'] = model.fit_predict(df.select_dtypes(include=[np.number]))
    anomalies = df[df['anomaly'] == -1]
    logging.info(f"Detected {len(anomalies)} anomalies.")
    df.drop(columns=['anomaly'], inplace=True)
    return df

# NLP for Metadata Interpretation

def interpret_metadata(metadata):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([metadata])
    logging.info(f"Metadata interpretation: {vectorizer.get_feature_names_out()}")

# User Feedback Mechanism

feedback_logs = []

def collect_user_feedback():
    feedback = input("Please provide feedback on the cleaning process: ")
    feedback_logs.append(feedback)
    logging.info("User feedback collected.")

def clean_dataset(df, meta_output):
    try:
        # Data Profiling
        profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
        profile.to_file("profiling_report.html")
        logging.info("Data profiling report generated.")

        # --- Process: Remove Duplicates ---
        df.drop_duplicates(inplace=True)
        logging.info("Removed duplicates.")

        # --- Process: Replace Empty Strings with NaN ---
        df.replace("", np.nan, inplace=True)
        logging.info("Replaced empty strings with NaN.")

        # --- Process: Convert Columns to DateTime ---
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().mean() >= 0.6:
                        logging.info(f"Converted {col} to DateTime.")
                except Exception as e:
                    logging.warning(f"Skipping {col} for DateTime conversion: {e}")

        # --- Process: Convert Columns to Numeric ---
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info(f"Converted {col} to Numeric.")
            except Exception as e:
                logging.warning(f"Skipping {col} for Numeric conversion: {e}")

        # --- Process: Fill Missing Values (numeric only) ---
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
            logging.info(f"Filled missing values in {col}.")

        # --- Process: Standardize Date Format to YYYY-MM-DD ---
        for col in df.select_dtypes(include=['datetime']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            logging.info(f"Standardized date format in {col}.")

        # --- Process: Handle Outliers using Isolation Forest (Explainable AI) ---
        for col in df.select_dtypes(include=[np.number]).columns:
            model = IsolationForest(contamination=0.1)
            df['outlier'] = model.fit_predict(df[[col]])
            df = df[df['outlier'] != -1]
            logging.info(f"Handled outliers in {col} using Isolation Forest.")
            logging.info(f"Feature importance: {model.feature_importances_}")
            df.drop(columns=['outlier'], inplace=True)

        # Feedback Loop: Log the cleaning process
        feedback_logs.append(f"Processed {df.shape[0]} rows with {df.shape[1]} columns.")

    except Exception as e:
        logging.error(f"An error occurred during cleaning: {e}")

    return df

def modify_dataset(df):
    # --- Process: Rename Columns ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logging.info("Renamed columns to snake_case.")

    # --- Process: Reorder Columns ---
    # Example: Move 'user_id' to the front
    if 'user_id' in df.columns:
        cols = ['user_id'] + [col for col in df.columns if col != 'user_id']
        df = df[cols]
        logging.info("Reordered columns to place 'user_id' first.")

    # --- Process: Drop Irrelevant Columns ---
    irrelevant_cols = ['temporary_column']  # Example
    df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)
    logging.info("Dropped irrelevant columns.")

    # --- Process: Filter Rows Based on Conditions ---
    df = df[df['active'] == True]  # Example condition
    logging.info("Filtered rows based on conditions.")

    # --- Process: Group and Aggregate ---
    if 'user_id' in df.columns and 'steps' in df.columns:
        df = df.groupby('user_id').agg({'steps': 'mean'}).reset_index()
        logging.info("Grouped by 'user_id' and aggregated 'steps'.")

    # --- Process: Merge or Join Datasets ---
    # Example: Merge with another DataFrame
    # df = df.merge(other_df, on='user_id', how='left')
    # logging.info("Merged datasets on 'user_id'.")

    # --- Process: Split or Combine Columns ---
    if 'full_name' in df.columns:
        df[['first_name', 'last_name']] = df['full_name'].str.split(' ', 1, expand=True)
        logging.info("Split 'full_name' into 'first_name' and 'last_name'.")

    return df

def transform_dataset(df):
    # --- Process: Normalization / Standardization ---
    scaler = StandardScaler()
    if 'steps' in df.columns:
        df['steps'] = scaler.fit_transform(df[['steps']])
        logging.info("Normalized 'steps'.")

    # --- Process: Encoding Categorical Values ---
    if 'mood' in df.columns:
        encoder = LabelEncoder()
        df['mood_encoded'] = encoder.fit_transform(df['mood'])
        logging.info("Encoded 'mood' as numeric.")

    # --- Process: Log Transformation ---
    if 'income' in df.columns:
        df['log_income'] = np.log1p(df['income'])
        logging.info("Applied log transformation to 'income'.")

    # --- Process: Create New Features ---
    if 'weight' in df.columns and 'height' in df.columns:
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
        logging.info("Created 'bmi' feature.")

    # --- Process: Binning / Bucketing ---
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 65, 100], labels=['child', 'adult', 'senior'])
        logging.info("Binned 'age' into categories.")

    # --- Process: Pivoting or Unpivoting Data ---
    # Example: Pivot data
    # df_pivot = df.pivot(index='date', columns='mood', values='steps')
    # logging.info("Pivoted data by 'mood'.")

    return df

def log_process(name, reason, code, impact):
    logging.info(f"--- Process: {name} ---")
    logging.info(f"Why: {reason}")
    logging.info(f"Code: {code}")
    logging.info(f"Impact: {impact}")

# Meta-Aware Cleaning

def meta_aware_cleaning(df, metadata):
    if 'timestamp' in metadata:
        for col in df.columns:
            if 'date' in col or 'time' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                log_process("Convert to DateTime", "Metadata indicates timestamp.", f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')", f"Converted {col} to DateTime.")
    return df

# Crew Workflow Automation

crew = Crew(agents=[cleaner_agent, modifier_agent, transformer_agent, summarizer_agent])

# Profiling Report Customization

def customized_profiling_report(df):
    profile = ProfileReport(df, title="Customized Data Profiling Report", explorative=True)
    profile.to_file("customized_profiling_report.html")
    logging.info("Customized profiling report generated.")

# Saving Logs as Report

def save_logs_as_report():
    with open("process_report.md", "w") as report_file:
        for log in feedback_logs:
            report_file.write(log + "\n")
    logging.info("Process report saved as process_report.md.")

def log_and_record(msg):
    logging.info(msg)
    feedback_logs.append(msg)

# Example usage of new features
if __name__ == "__main__":
    # Get input files
    file_paths, meta_output = get_input_files()

    # Process each dataset
    for i, file_path in enumerate(file_paths):
        logging.info(f"Processing dataset {i+1}...")
        try:
            df = pd.read_csv(file_path)
            cleaned_df = clean_dataset(df, meta_output)
            
            with open(meta_output, "r") as file:
                metadata_text = file.read()

            meta_cleaned_df = meta_aware_cleaning(cleaned_df, metadata_text)
            modified_df = modify_dataset(meta_cleaned_df)
            transformed_df = transform_dataset(modified_df)
            selected_df = select_features(transformed_df, target_column='target')  # Example target column
            anomaly_free_df = detect_anomalies(selected_df)
            interpret_metadata(metadata_text)
            customized_profiling_report(anomaly_free_df)
            output_path = f"final_dataset_{i+1}.csv"
            anomaly_free_df.to_csv(output_path, index=False)
            logging.info(f"Final data saved to: {output_path}")
            collect_user_feedback()
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    # Run Crew Workflow
    crew.run([cleaning_task, modifying_task, transforming_task, summary_task])

    # Output feedback logs
    save_logs_as_report()
