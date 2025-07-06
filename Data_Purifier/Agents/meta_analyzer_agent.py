# agents/meta_analyzer_agent.py
# This agent is responsible for analyzing dataset metadata, loading raw data,
# and preparing it for the data purification pipeline. It uses an LLM to parse
# meta-information and handles internal data format conversions (e.g., CSV to Feather).

import json
import logging
import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor # Import for parallel processing

import pandas as pd
import spacy # For Natural Language Processing (NLP) tasks
# Tenacity is used for retrying operations that might fail due to transient issues (e.g., LLM calls)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class MetaAnalyzerAgent:
    """
    Agent responsible for initial data analysis, schema inference, and data loading.
    It leverages an LLM to interpret meta-information and prepares the DataFrame
    for subsequent cleaning, modification, and transformation stages.
    """

    def __init__(self, config: Optional[Dict] = None, llm=None):
        """
        Initializes the MetaAnalyzerAgent.

        Args:
            config (Optional[Dict]): Configuration dictionary, typically from config/settings.py.
            llm: An optional Language Model instance. If not provided, it will be initialized
                 based on the configuration.
        """
        self.logger = logging.getLogger(__name__)
        # Default LLM configuration if not provided
        self.config = config or {
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "llm_temperature": 0.7
        }
        # Initialize the LLM based on the provided configuration
        self.llm = self._initialize_llm(self.config)

        # Initialize spaCy for NLP tasks (e.g., identifying relevant columns from questions)
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            # If the spaCy model is not found, download it
            print("Downloading spacy model...")
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        self.analysis_report = {} # Stores the analysis report
        self.cache_dir = "./.meta_analyzer_cache" # Directory for caching LLM responses and Feather files
        os.makedirs(self.cache_dir, exist_ok=True) # Ensure cache directory exists

    def _initialize_llm(self, config: Dict):
        """
        Initializes the Language Model based on the provider specified in the configuration.

        Args:
            config (Dict): Configuration dictionary containing LLM provider, model, and API keys.

        Returns:
            An initialized LLM instance (e.g., ChatOpenAI, ChatGoogleGenerativeAI).

        Raises:
            ValueError: If an unsupported LLM provider is specified or API key is missing.
        """
        llm_provider = config.get("llm_provider")
        llm_model = config.get("llm_model")
        llm_temperature = config.get("llm_temperature")

        if llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = config.get("openai_api_key")
            # For testing purposes, a mock LLM is used if OPENAI_API_KEY is not set.
            # In a production environment, ensure the API key is properly configured.
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not set. Using a mock LLM for testing/development.")
                from unittest.mock import Mock
                mock_llm = Mock()
                # Mock response structure for schema, questions, pipeline, and operations
                mock_llm.invoke.return_value.content = "{ \"schema\": { \"columns\": [], \"rows\": 0, \"column_types\": {} }, \"questions\": [], \"pipeline_plan\": [], \"suggested_operations\": { \"cleaning_operations\": {}, \"modification_operations\": {}, \"transformation_operations\": {} } }"
                return mock_llm
            return ChatOpenAI(temperature=llm_temperature, model_name=llm_model, api_key=api_key)
        elif llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = config.get("google_api_key")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in config for Google provider.")
            return ChatGoogleGenerativeAI(temperature=llm_temperature, model=llm_model, google_api_key=api_key)
        elif llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = config.get("anthropic_api_key")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in config for Anthropic provider.")
            return ChatAnthropic(temperature=llm_temperature, model_name=llm_model, anthropic_api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def analyze(self, num_datasets: int, dataset_paths: List[str], meta_output_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyzes, standardizes, and filters datasets based on metadata and LLM-parsed instructions.
        This is the main entry point for the MetaAnalyzerAgent's operations.

        Args:
            num_datasets (int): The total count of dataset files to process.
            dataset_paths (List[str]): A list of absolute paths to the raw data files (e.g., CSVs).
            meta_output_path (str): The absolute path to the metadata file (e.g., a text file with user questions).

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing:
                - The standardized and filtered DataFrame.
                - A comprehensive report of the analysis, including schema, questions, and suggested operations.

        Raises:
            FileNotFoundError: If the meta output file does not exist.
        """
        self.logger.info(f"Starting dataset analysis for {num_datasets} dataset(s).")

        # Ensure the meta output file exists before proceeding
        if not os.path.exists(meta_output_path):
            self.logger.error(f"Meta output file not found at: {meta_output_path}")
            raise FileNotFoundError(f"Meta output file not found at: {meta_output_path}")

        # --- 1. Parse metadata using LLM ---
        # The LLM interprets the meta_output.txt to extract structured information.
        self.logger.info(f"Parsing metadata from {meta_output_path} using LLM...")
        meta_info = self._parse_meta_output_with_llm(meta_output_path)

        # --- 2. Load and Validate Datasets ---
        # This step handles reading the raw data, converting CSVs to Feather internally,
        # and performing basic validation against the schema parsed by the LLM.
        self.logger.info(f"Loading datasets from: {', '.join(dataset_paths)}")
        df = self._load_and_validate_datasets(dataset_paths, meta_info)

        # --- 3. Standardize and Filter ---
        # Column names are standardized, data types are enforced, and irrelevant
        # columns are filtered based on the analytical questions.
        self.logger.info("Standardizing and filtering dataset based on metadata.")
        df, report = self._standardize_and_filter(df, meta_info)

        # Combine the LLM-parsed meta_info with the standardization/filtering report
        combined_report = {
            **meta_info, # Includes schema, questions, pipeline_plan, suggested_operations
            "analysis_report": report # Includes removed columns, data type changes
        }
        self.logger.info("Dataset analysis complete.")

        return df, combined_report

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(Exception))
    def _parse_meta_output_with_llm(self, meta_output_path: str) -> Dict:
        """
        Parses the content of the meta output file using an LLM to extract
        schema, analytical questions, pipeline plan, and suggested operations.
        Includes caching mechanism to avoid re-calling LLM for same content.

        Args:
            meta_output_path (str): The absolute path to the meta output file.

        Returns:
            Dict: A dictionary containing the structured meta-information.

        Raises:
            ValueError: If the LLM does not return valid JSON.
        """
        with open(meta_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Generate a hash of the file content to use as a cache key
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        cache_file_path = os.path.join(self.cache_dir, f"{content_hash}.json")

        # Check if the LLM response for this content is already cached
        if os.path.exists(cache_file_path):
            self.logger.info(f"Loading LLM response from cache: {cache_file_path}")
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Construct the prompt for the LLM, instructing it to return a specific JSON structure
        prompt = f"""You are a highly experienced Data Architect and Strategic Planner. Your core responsibility is to meticulously analyze raw dataset metadata and user-defined analytical questions to formulate a comprehensive, executable data purification strategy. Your output must be a perfectly structured JSON object, serving as the blueprint for the entire data pipeline. Precision and foresight are paramount.

JSON Output Specification:
- 'schema': A detailed representation of the dataset's structure after initial assessment. This must include:
  - 'columns': A definitive list of all relevant column names (strings).
  - 'rows': The precise total count of rows (integer).
  - 'column_types': An object mapping each column name to its most appropriate inferred data type (e.g., "string", "integer", "float", "boolean", "datetime"). Accuracy in type inference is crucial for downstream operations.
- 'questions': A concise list of the primary analytical questions or objectives derived directly from the user's input, guiding the purification process.
- 'pipeline_plan': An optimized, sequential list of data processing stages. The order must reflect logical dependencies and best practices for data quality. For example, cleaning typically precedes transformation. Provide a clear, concise justification for the chosen order of stages.
- 'suggested_operations': A granular breakdown of recommended operations for each stage. This object must contain three top-level keys, each mapping to an object where keys are standardized column names and values are lists of specific operations.
  - 'cleaning_operations': Focus on rectifying data quality anomalies. For each operation, provide a detailed 'reason' explaining *why* this operation is necessary for the specific column, considering data characteristics (e.g., distribution, presence of outliers/missing values). Examples: handling missing values, outlier detection and treatment, duplicate removal, and inconsistency resolution.
  - 'modification_operations': Focus on enhancing data for analytical purposes. For each operation, provide a detailed 'reason' explaining the analytical benefit or purpose of the modification. Examples: feature engineering (creating new variables), data aggregation, scaling/normalization, and discretization/binning.
  - 'transformation_operations': Focus on reshaping and converting data formats. For each operation, provide a detailed 'reason' explaining the necessity for data structure or format change. Examples: pivoting/unpivoting, merging/joining datasets, text processing (e.g., tokenization, stemming), and categorical encoding.

Each individual operation within 'suggested_operations' must adhere to this exact structure:
{{
  "column_name": [
    {"operation": "operation_type_string", "method": "specific_method_string (if applicable, e.g., 'median', 'one_hot')", "reason": "detailed_and_clear_justification_string"}
  ]
}}

Crucial Formatting Rule: All column names within 'suggested_operations' must be converted to `snake_case` (lowercase, words separated by underscores). For instance, 'Customer ID' becomes 'customer_id'.

Raw Metadata and User Questions for Analysis:
'''
{content}
'''

Your output must be ONLY the JSON object. No preamble, no conversational filler, no explanations outside the JSON structure. Ensure the JSON is valid and strictly follows the schema.

Example of 'suggested_operations' structure for clarity:
{{
  "pipeline_plan": ["cleaning", "transformation", "modification"],
  "cleaning_operations": {{
    "customer_id": [
      {{"operation": "remove_duplicates", "method": "all", "reason": "To ensure each customer record is unique, preventing skewed analysis results due to redundant entries."}}
    ],
    "transaction_amount": [
      {{"operation": "handle_outliers", "method": "iqr_clipping", "reason": "To mitigate the impact of extreme transaction values that could disproportionately influence statistical measures and model training, ensuring a more robust dataset."}}
    ]
  }},
  "modification_operations": {{
    "order_date": [
      {{"operation": "feature_engineering", "method": "extract_month", "reason": "To derive a 'month' feature from the order date, enabling analysis of monthly sales trends and seasonality."}}
    ]
  }},
  "transformation_operations": {{
    "product_category": [
      {{"operation": "categorical_encoding", "method": "one_hot", "reason": "To convert categorical product categories into a numerical format suitable for machine learning algorithms, which typically require numerical input."}}
    ]
  }}
}}
"""

        try:
            llm_response = self.llm.invoke(prompt)
            # Attempt to parse the LLM's response as JSON
            meta_info = json.loads(llm_response.content)

            # Save the LLM's structured response to cache for future use
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2)
            self.logger.info(f"Saved LLM response to cache: {cache_file_path}")

            self.logger.info(f"LLM parsed meta info: {meta_info}")
            return meta_info
        except json.JSONDecodeError as e:
            # Handle cases where the LLM does not return valid JSON
            self.logger.error(f"Failed to decode JSON from LLM response: {e}. Response: {llm_response.content}")
            raise ValueError("LLM did not return valid JSON. Check LLM prompt and response format.")
        except Exception as e:
            # Catch any other exceptions during LLM parsing
            self.logger.error(f"Error during LLM parsing of meta output: {e}", exc_info=True)
            raise

    def _load_and_validate_datasets(self, dataset_paths: List[str], meta_info: Dict) -> pd.DataFrame:
        """
        Loads datasets from the provided paths, converts CSVs to Feather format internally
        for efficiency, and performs basic validation against the schema from the meta file.

        Args:
            dataset_paths (List[str]): A list of absolute paths to the raw data files.
            meta_info (Dict): The structured meta-information parsed by the LLM.

        Returns:
            pd.DataFrame: A combined DataFrame from all loaded datasets.

        Raises:
            ValueError: If no dataset paths are provided or no valid data can be loaded.
        """
        if not dataset_paths:
            raise ValueError("No dataset paths provided.")

        processed_dfs = []
        # Use ThreadPoolExecutor for parallel loading of datasets
        with ThreadPoolExecutor() as executor:
            # Submit loading tasks for each dataset path
            futures = [executor.submit(self._load_single_dataset, path) for path in dataset_paths]
            for future in futures:
                df = future.result()
                if df is not None:
                    processed_dfs.append(df)

        if not processed_dfs:
            raise ValueError("Could not load any valid data from the provided paths. Please check file paths and formats.")

        # Concatenate all loaded DataFrames into a single DataFrame for unified processing
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        self.logger.info(f"Duplicates after initial load and concat: {combined_df.duplicated().sum()}")
        # Force unique rows to handle any subtle duplicates introduced during loading
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        self.logger.info(f"Duplicates after forcing unique rows: {combined_df.duplicated().sum()}")

        # Basic validation against the schema parsed by the LLM
        expected_cols = meta_info.get('schema', {}).get('columns', [])
        if expected_cols and not all(col in combined_df.columns for col in expected_cols):
            self.logger.warning(f"Column mismatch detected. Expected columns from meta-info: {expected_cols}, Found in combined DataFrame: {list(combined_df.columns)}. Proceeding, but data integrity might be affected.")

        return combined_df

    def _load_single_dataset(self, original_path: str) -> Optional[pd.DataFrame]:
        """
        Helper function to load a single dataset, convert CSV to Feather if necessary,
        and return the DataFrame.
        """
        if not os.path.exists(original_path):
            self.logger.warning(f"Dataset file not found: {original_path}. Skipping.")
            return None

        file_extension = os.path.splitext(original_path)[1].lower()

        try:
            if file_extension == '.csv':
                self.logger.info(f"Loading CSV: {original_path} and converting to Feather for internal use.")
                df_to_load = pd.read_csv(original_path)
                feather_path = os.path.join(self.cache_dir, os.path.basename(original_path).replace('.csv', '.feather'))
                df_to_load.to_feather(feather_path)
                self.logger.info(f"Converted {original_path} to {feather_path} and cached.")
                return pd.read_feather(feather_path)
            elif file_extension == '.feather' or file_extension == '.parquet':
                self.logger.info(f"Loading {file_extension.upper()}: {original_path}.")
                return pd.read_feather(original_path) if file_extension == '.feather' else pd.read_parquet(original_path)
            else:
                self.logger.error(f"Unsupported file format for {original_path}. Only .csv, .feather, .parquet are supported. Skipping.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load or convert dataset from {original_path}: {e}", exc_info=True)
            return None

    def _standardize_and_filter(self, df: pd.DataFrame, meta_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardizes column names, enforces data types, and filters columns
        based on the schema and analytical questions provided in the meta-information.

        Args:
            df (pd.DataFrame): The input DataFrame.
            meta_info (Dict): The structured meta-information parsed by the LLM.

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing:
                - The standardized and filtered DataFrame.
                - A report detailing changes (removed columns, data type changes).
        """
        report = {
            "removed_columns": [],
            "data_type_changes": []
        }

        original_columns = df.columns.copy() # Keep track of original column names

        # --- Standardize column names ---
        # Convert all column names to snake_case for consistency
        df.columns = [self._standardize_column_name(col) for col in df.columns]
        # Create a mapping from new standardized names back to original names (for reporting/debugging)
        standardized_map = dict(zip(df.columns, original_columns))

        self.logger.info(f"Duplicates before standardization/filtering in MetaAnalyzerAgent: {df.duplicated().sum()}")

        # --- Standardize data types ---
        # Enforce data types based on the 'column_types' specified in the meta-information
        expected_types = meta_info.get('schema', {}).get('column_types', {})
        for col_name, expected_dtype in expected_types.items():
            standardized_col_name = self._standardize_column_name(col_name) # Get the standardized name
            if standardized_col_name in df.columns:
                original_dtype = df[standardized_col_name].dtype
                try:
                    target_dtype = self._map_dtype(expected_dtype) # Map string type to pandas dtype
                    if target_dtype and original_dtype != target_dtype: # If a valid target type and different
                        df[standardized_col_name] = df[standardized_col_name].astype(target_dtype) # Perform type conversion
                        report['data_type_changes'].append({
                            "column": standardized_col_name,
                            "from": str(original_dtype),
                            "to": str(df[standardized_col_name].dtype)
                        }) # Log the change
                except Exception as e:
                    self.logger.error(f"Could not convert column '{col_name}' (standardized to '{standardized_col_name}') to {expected_dtype}: {e}", exc_info=True)

        # --- Filter columns based on analytical questions ---
        # This step aims to keep only columns relevant to the user's questions.
        questions = meta_info.get('questions', [])
        if questions: # Only filter if questions are provided
            all_question_text = ' '.join(questions).lower() # Combine all questions into a single string

            # Use spaCy's NLP capabilities to extract potential column names (nouns, proper nouns)
            doc = self.nlp(all_question_text)
            necessary_columns = {token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN']} # Extract lemmas of nouns/proper nouns

            # Also include column names that are explicitly mentioned in the questions
            for col in df.columns:
                 if col.lower() in all_question_text:
                    necessary_columns.add(col)

            # Determine which columns to keep based on the identified necessary columns
            columns_to_keep = {col for col in df.columns if col in necessary_columns}

            if not columns_to_keep: # Fallback: if no relevant columns are identified, keep all
                self.logger.warning("NLP analysis did not identify any necessary columns based on questions. Keeping all columns.")
                columns_to_keep = set(df.columns)

            # Identify and log removed columns
            removed_cols = set(df.columns) - columns_to_keep
            report['removed_columns'] = list(removed_cols)
            df = df[list(columns_to_keep)] # Filter the DataFrame to keep only relevant columns

        return df, report

    def _standardize_column_name(self, column: str) -> str:
        """
        Standardizes a column name to `snake_case` (lowercase with underscores).
        Removes special characters and replaces spaces with underscores.

        Args:
            column (str): The original column name.

        Returns:
            str: The standardized column name.
        """
        if not isinstance(column, str):
            column = str(column) # Ensure it's a string
        column = column.strip().lower() # Remove leading/trailing whitespace and convert to lowercase
        column = re.sub(r'[^a-z0-9_]+', '_', column) # Replace non-alphanumeric (except underscore) with underscore
        return column

    def _map_dtype(self, dtype_str: str):
        """
        Maps a string representation of a data type (from LLM output) to a Pandas/NumPy dtype.

        Args:
            dtype_str (str): The string representation of the data type (e.g., "integer", "datetime").

        Returns:
            Any: The corresponding Pandas/NumPy dtype, or None if not recognized.
        """
        dtype_map = {
            'string': 'object', # Python object type, typically for strings
            'str': 'object',
            'integer': 'int64', # 64-bit integer
            'int': 'int64',
            'float': 'float64', # 64-bit floating point
            'double': 'float64',
            'boolean': 'bool', # Boolean type
            'bool': 'bool',
            'datetime': 'datetime64[ns]', # Nanosecond precision datetime
            'date': 'datetime64[ns]'
        }
        return dtype_map.get(dtype_str.lower(), None) # Return mapped dtype or None if not found
