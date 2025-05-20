import json
import pandas as pd
import os

def get_input_files():
    num_datasets = int(input("Enter the number of datasets: "))
    dataset_paths = [
        input(f"Enter the file path for dataset {i+1}: ")
        for i in range(num_datasets)
    ]
    meta_output_path = input("Enter the path for meta output file (or press Enter for default): ")
    if not meta_output_path:
        meta_output_path = os.path.join(os.path.dirname(__file__), "meta_output.txt")
    return dataset_paths, meta_output_path

def save_logs_as_report(feedback_logs):
    with open("process_report.md", "w") as report_file:
        for log in feedback_logs:
            report_file.write(log + "\n")

def save_json_report(report_data, output_dir, filename="processing_report.json"):
    """Saves the report data as a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    report_path = os.path.join(output_dir, filename)
    try:
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        print(f"Successfully saved JSON report to {report_path}")
        return report_path
    except Exception as e:
        print(f"Error saving JSON report: {e}")
        return None

def save_dataframe_as_csv(df, output_dir, filename="cleaned_dataset.csv"):
    """Saves the pandas DataFrame as a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, filename)
    try:
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved DataFrame to {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")
        return None