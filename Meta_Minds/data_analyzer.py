# This file contains functions for analyzing datasets, specifically for generating column descriptions using an AI model.

from config import client # Import the pre-initialized OpenAI client
import pandas as pd # Needed for DataFrame type hinting and operations
import logging

def generate_column_descriptions(data: pd.DataFrame) -> dict:
    """Generates descriptions for each column in a DataFrame using GPT.

    This function iterates through the columns of a pandas DataFrame, extracts sample values,
    and uses an OpenAI model (via the imported client) to generate a short description
    for each column based on its name and sample data.

    Args:
        data (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary mapping column names (str) to their generated descriptions (str).
              Returns an empty dictionary if the input DataFrame is empty or if description
              generation fails for all columns.
    """
    """Generates descriptions for each column in a DataFrame using GPT.

    Args:
        data (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary mapping column names to their descriptions.
    """
    logging.info("Generating column descriptions using GPT...")
    descriptions = {}
    # Check if data is empty before proceeding
    if data.empty:
        logging.warning("DataFrame is empty, cannot generate column descriptions.")
        return descriptions

    for column in data.columns:
        # Get sample values, handling potential errors
        try:
            sample_values = data[column].dropna().astype(str).head(3).tolist()
            sample_text = ", ".join(sample_values)
            if not sample_text:
                 sample_text = "[No non-null values]" # Handle columns with all nulls
        except Exception as e:
            logging.warning(f"Could not get sample values for column '{column}': {e}")
            sample_text = "[Could not extract samples]"


        prompt = (
            f"Column name: {column}\n"
            f"Sample values: {sample_text}\n"
            f"Based on the column name and sample values, write a short, clear description of this column:"
        )
        try:
            # Use the client imported from config
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Consider making model configurable via config.py
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5 # Add temperature for slightly varied descriptions if needed
            )
            description = response.choices[0].message.content.strip()
            descriptions[column] = description
            logging.debug(f"Generated description for '{column}': {description}")
        except Exception as e:
            logging.error(f"Failed to generate description for '{column}' using GPT: {e}")
            # Provide a fallback description
            descriptions[column] = f"Description failed. Sample values: {sample_text}"

    logging.info(f"Finished generating descriptions for {len(descriptions)} columns.")
    return descriptions


def generate_summary(df: pd.DataFrame) -> dict:
    """Generates a summary of the DataFrame, including row count, column count, and column descriptions.

    This function calculates the number of rows and columns and calls `generate_column_descriptions`
    to get AI-generated descriptions for each column. It then compiles this information
    into a structured dictionary.

    Args:
        df (pandas.DataFrame): The DataFrame to summarize.

    Returns:
        dict: A dictionary containing the summary information:
              - 'rows' (int): Number of rows.
              - 'columns' (int): Number of columns.
              - 'column_info' (dict): A dictionary mapping column names to their dtype and description.
              Returns a summary with zero counts and empty column info if the input DataFrame is empty.
    """
    logging.info("Generating data summary...")
    if df.empty:
        logging.warning("DataFrame is empty, generating empty summary.")
        return {
            "rows": 0,
            "columns": 0,
            "column_info": {}
        }

    meta = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": {}
    }

    # This calls the GPT description function
    descriptions = generate_column_descriptions(df)

    for col in df.columns:
        meta["column_info"][col] = {
            "dtype": str(df[col].dtype),
            # Use the description from the GPT call, or a fallback if generation failed for this column
            "description": descriptions.get(col, "Description unavailable")
        }
    logging.info("Finished generating data summary.")
    return meta
