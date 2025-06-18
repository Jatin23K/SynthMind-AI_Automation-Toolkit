import pandas as pd
import os
import logging

def read_file(file_path):
    """Loads data from a CSV, Excel, or JSON file.

    Args:
        file_path (str): The path to the data file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported.
        Exception: For other errors during file reading (e.g., corrupted file).
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        logging.info(f"Attempting to read file: {file_path} with extension: {ext}")
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            logging.error(f"Unsupported file type: {ext} for file: {file_path}")
            raise ValueError(f"Unsupported file type: {ext}")

        logging.info(f"Successfully read file: {file_path}. Shape: {df.shape}")
        if df.empty:
            logging.warning(f"File {file_path} is empty.")
            # Depending on requirements, you might want to raise an error for empty files
            # raise ValueError(f"File {file_path} is empty.")
        return df

    except FileNotFoundError:
        logging.error(f"Error: File not found at path: {file_path}")
        raise # Re-raise the exception after logging
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading file {file_path}: {e}")
        raise # Re-raise the exception after logging