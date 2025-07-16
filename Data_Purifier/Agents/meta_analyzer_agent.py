# agents/meta_analyzer_agent.py
# This agent is responsible for analyzing metadata, loading raw data,
# and preparing it for the data purification pipeline.

import logging
import pandas as pd
import yaml # Import the YAML library for parsing meta_output.txt
from typing import Dict, List, Tuple, Any # Import Any for flexible type hinting

from data_purifier.utils.cached_chat_openai import CachedChatOpenAI # Import LLM for intelligent column selection

class MetaAnalyzerAgent:
    """
    The MetaAnalyzerAgent analyzes metadata, loads raw data, and prepares it for processing.
    It also handles the internal conversion of CSV to Feather for efficiency.
    """

    def __init__(self, config: Dict):
        """
        Initializes the MetaAnalyzerAgent.

        Args:
            config (Dict): Configuration dictionary for the system.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = CachedChatOpenAI(temperature=self.config.get("llm_temperature", 0.7), model_name=self.config.get("llm_model", "gpt-4"))

    def _parse_meta_output(self, meta_output_path: str) -> Dict:
        """
        Parses the meta_output.txt file (expected to be in YAML format)
        to extract schema, questions, and suggested operations.
        """
        try:
            with open(meta_output_path, 'r') as f:
                meta_content = yaml.safe_load(f)
            if isinstance(meta_content, dict):
                return meta_content
            elif meta_content is None:
                return {}
            else:
                self.logger.warning(f"Meta output file {meta_output_path} contained non-dictionary YAML: {meta_content}. Returning empty dict.")
                return {}
        except FileNotFoundError:
            self.logger.warning(f"Meta output file not found: {meta_output_path}. Proceeding without meta-instructions.")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML in meta output file {meta_output_path}: {e}")
            return {}

    def _get_relevant_columns_from_llm(self, columns: List[str], questions: List[str]) -> List[str]:
        """
        Interacts with the LLM to determine which columns are relevant to the given questions.
        """
        if not questions:
            self.logger.info("No questions provided in meta_output.txt. All columns considered relevant.")
            return columns

        column_list_str = ", ".join(columns)
        question_list_str = "\n".join([f"- {q}" for q in questions])

        prompt = f"""
        Given the following DataFrame columns: {column_list_str}
        And the following analytical questions:
        {question_list_str}

        Identify and list ONLY the column names that are directly relevant to answering these questions.
        Return the column names as a comma-separated string. If no columns are relevant, return "None".
        Example: "column1, column2, column3"
        """
        try:
            # Use the LLM to get relevant columns
            response = self.llm.invoke(prompt)
            relevant_cols_str = response.content.strip()

            if relevant_cols_str.lower() == "none" or not relevant_cols_str:
                return []
            
            # Parse the comma-separated string into a list
            relevant_cols = [col.strip() for col in relevant_cols_str.split(',')]
            self.logger.info(f"LLM identified relevant columns: {relevant_cols}")
            return relevant_cols
        except Exception as e:
            self.logger.error(f"Error interacting with LLM for column relevance: {e}", exc_info=True)
            self.logger.warning("Falling back to considering all columns relevant due to LLM error.")
            return columns

    def analyze(self, num_datasets: int, dataset_paths: List[str], meta_output_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyzes metadata, loads raw data, and prepares it for processing.
        This method performs data loading, type optimization, and generates a comprehensive
        meta-analysis report with suggested cleaning, modification, and transformation operations
        based on the characteristics of the loaded data.
        
        Args:
            num_datasets (int): The number of datasets to analyze. (Currently expects 1 per call for simplicity).
            dataset_paths (List[str]): A list of absolute paths to the input datasets (CSV files).
            meta_output_path (str): The absolute path to the meta output file. This file is currently
                                    used for placeholder output, but in a full LLM integration, it would
                                    contain detailed instructions from an LLM.

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing:
                - pd.DataFrame: The concatenated and optimized DataFrame of all input datasets.
                - Dict: A meta-analysis report containing suggested operations for the pipeline,
                        analysis findings, and schema information.
        """
        self.logger.info(f"MetaAnalyzerAgent: Analyzing {num_datasets} datasets from {dataset_paths}")

        all_dfs = []
        for path in dataset_paths:
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
                self.logger.info(f"Successfully loaded {path}. Shape: {df.shape}")
            except Exception as e:
                self.logger.error(f"Error loading dataset from {path}: {e}", exc_info=True)
                raise # Re-raise the exception to halt processing if data loading fails.

        if not all_dfs:
            raise ValueError("No datasets were successfully loaded.")

        # Concatenate all loaded DataFrames into a single DataFrame.
        # `ignore_index=True` resets the index of the combined DataFrame.
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"Combined DataFrame shape: {combined_df.shape}")

        # Parse meta_output.txt for schema, questions, and initial suggestions
        meta_info = self._parse_meta_output(meta_output_path)
        parsed_questions = meta_info.get('questions', [])
        parsed_schema_column_types = meta_info.get('schema', {}).get('column_types', {})

        # Apply data type standardization based on meta_output.txt if specified
        for col, dtype_name in parsed_schema_column_types.items():
            if col in combined_df.columns:
                try:
                    if dtype_name == 'datetime':
                        combined_df[col] = pd.to_datetime(combined_df[col])
                    elif dtype_name == 'integer':
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').astype(pd.Int64Dtype())
                    elif dtype_name == 'float':
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                    elif dtype_name == 'boolean':
                        combined_df[col] = combined_df[col].astype(bool)
                    # Add more type conversions as needed
                    self.logger.info(f"Standardized column '{col}' to type '{dtype_name}' based on meta_output.txt.")
                except Exception as e:
                    self.logger.warning(f"Could not standardize column '{col}' to type '{dtype_name}': {e}")

        # Optimize DataFrame data types for memory efficiency and performance.
        optimized_df = self._optimize_dataframe_types(combined_df)
        self.logger.info(f"Optimized DataFrame shape: {optimized_df.shape}, memory usage: {optimized_df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

        # Determine relevant columns using LLM based on questions
        all_current_columns = optimized_df.columns.tolist()
        relevant_columns = self._get_relevant_columns_from_llm(all_current_columns, parsed_questions)

        # Initialize dictionaries to store suggested operations and analysis findings.
        suggested_operations = meta_info.get('suggested_operations', {
            "cleaning_operations": {},
            "modification_operations": {},
            "transformation_operations": {}
        })
        analysis_report = {
            "removed_columns": [],
            "data_type_changes": []
        }

        # Add drop operations for irrelevant columns
        columns_to_drop = [col for col in all_current_columns if col not in relevant_columns]
        if columns_to_drop:
            if "global" not in suggested_operations["modification_operations"]:
                suggested_operations["modification_operations"]["global"] = []
            for col in columns_to_drop:
                suggested_operations["modification_operations"]["global"].append({
                    "operation": "drop_column",
                    "column_name": col,
                    "reason": f"Column '{col}' identified as not relevant to analytical questions by LLM."
                })
                analysis_report["removed_columns"].append(col)
            self.logger.info(f"Added drop operations for irrelevant columns: {columns_to_drop}")
        
        # Filter the DataFrame to keep only relevant columns before further analysis
        if relevant_columns:
            optimized_df = optimized_df[relevant_columns]
            self.logger.info(f"DataFrame filtered to relevant columns. New shape: {optimized_df.shape}")
        else:
            self.logger.warning("No relevant columns identified by LLM or questions. Proceeding with all columns.")


        # Iterate through each column of the optimized DataFrame to suggest operations.
        for col in optimized_df.columns:
            # --- Suggest Missing Value Handling ---
            if optimized_df[col].isnull().any(): # Check if the column contains any missing values.
                missing_count = optimized_df[col].isnull().sum()
                if pd.api.types.is_numeric_dtype(optimized_df[col]):
                    # For numeric columns, suggest median imputation (more robust to outliers).
                    suggested_operations["cleaning_operations"].setdefault(col, []).append({
                        "operation": "handle_missing_values",
                        "method": "median", 
                        "reason": f"Column has {missing_count} missing values. Median imputation suggested."
                    })
                else:
                    # For non-numeric (categorical/object) columns, suggest mode imputation.
                    suggested_operations["cleaning_operations"].setdefault(col, []).append({
                        "operation": "handle_missing_values",
                        "method": "mode",
                        "reason": f"Column has {missing_count} missing values. Mode imputation suggested."
                    })

            # --- Suggest Outlier Handling (for numeric columns only) ---
            if pd.api.types.is_numeric_dtype(optimized_df[col]):
                # Heuristic for outlier suggestion: if there are missing values (which might be outliers)
                # or if the standard deviation is significantly high compared to the mean (indicating spread).
                # In a real-world scenario, a more sophisticated outlier detection algorithm would be used here.
                if optimized_df[col].isnull().any() or optimized_df[col].std() > optimized_df[col].mean() * 0.5: 
                    suggested_operations["cleaning_operations"].setdefault(col, []).append({
                        "operation": "handle_outliers",
                        "method": "isolation_forest", # Suggest Isolation Forest as a general-purpose outlier detection method.
                        "reason": "Numeric column, consider outlier handling."
                    })

            # --- Suggest Categorical Encoding / Inconsistency Handling ---
            # Applies to object (string) or categorical dtype columns.
            if pd.api.types.is_object_dtype(optimized_df[col]) or pd.api.types.is_categorical_dtype(optimized_df[col]):
                unique_ratio = optimized_df[col].nunique() / len(optimized_df) # Ratio of unique values to total rows.
                
                # If low cardinality (e.g., <10% unique values) and more than binary, suggest one-hot encoding.
                if unique_ratio < 0.1 and optimized_df[col].nunique() > 2: 
                    suggested_operations["transformation_operations"].setdefault(col, []).append({
                        "operation": "encode_categorical",
                        "method": "one_hot",
                        "reason": "Low cardinality categorical column, suitable for one-hot encoding."
                    })
                # If medium to high cardinality (e.g., <50% unique values but more than 50 unique values),
                # suggest frequency encoding to reduce dimensionality while retaining information.
                elif unique_ratio < 0.5 and optimized_df[col].nunique() > 50: 
                    suggested_operations["transformation_operations"].setdefault(col, []).append({
                        "operation": "encode_categorical",
                        "method": "frequency_encode",
                        "reason": "Medium to high cardinality categorical column, frequency encoding suggested."
                    })
                
                # Always suggest inconsistency handling for object/categorical columns due to potential typos or variations.
                suggested_operations["cleaning_operations"].setdefault(col, []).append({
                    "operation": "handle_inconsistencies",
                    "method": "fuzzy_match", # Fuzzy matching can help standardize similar string entries.
                    "reason": "Categorical column, consider handling inconsistencies."
                })

        # Add global operations like duplicate removal, which applies to the entire dataset.
        suggested_operations["cleaning_operations"].setdefault("global", []).append({
            "operation": "remove_duplicates",
            "reason": "Initial duplicate check for the entire dataset."
        })

        # Construct the final meta-analysis report.
        meta_analysis_report = {
            "pipeline_plan": meta_info.get('pipeline_plan', ["cleaning", "modification", "transformation"]), # Use parsed plan or default
            "suggested_operations": suggested_operations, # Include all dynamically suggested operations.
            "analysis_report": analysis_report, # Include general analysis findings.
            "questions": parsed_questions, # Include parsed questions
            "schema": { # Include schema information, specifically column types.
                "column_types": optimized_df.dtypes.apply(lambda x: x.name).to_dict()
            }
        }
        self.logger.info("Generated comprehensive meta-analysis report based on data characteristics.")

        # In a real scenario, this agent would interact with an LLM to refine instructions
        # or generate more complex insights. For now, it writes a placeholder to the meta_output_path.
        try:
            with open(meta_output_path, 'w') as f:
                # Write the full meta_analysis_report to the meta_output_path for debugging/inspection
                yaml.dump(meta_analysis_report, f, default_flow_style=False)
            self.logger.info(f"Wrote meta-analysis report to {meta_output_path}")
        except Exception as e:
            self.logger.error(f"Error writing meta-analysis output to {meta_output_path}: {e}", exc_info=True)

        return optimized_df, meta_analysis_report

    def _optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes DataFrame column data types to reduce memory usage and improve performance.
        This involves downcasting numeric types (integers, floats) to their smallest possible
        representation and converting suitable object columns to categorical types.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with optimized data types.
        """
        optimized_df = df.copy() # Work on a copy to avoid modifying the original DataFrame.
        for col in optimized_df.columns:
            # --- Optimize Numeric Types ---
            if pd.api.types.is_numeric_dtype(optimized_df[col]):
                try:
                    # Attempt to downcast integer types (e.g., from int64 to int32 or int16).
                    if pd.api.types.is_integer_dtype(optimized_df[col]):
                        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                    # Attempt to downcast float types (e.g., from float64 to float32).
                    elif pd.api.types.is_float_dtype(optimized_df[col]):
                        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                except Exception as e:
                    # Log a debug message if downcasting fails (e.g., due to overflow).
                    self.logger.debug(f"Could not downcast numeric column {col}: {e}")

            # --- Convert Object Columns to Categorical if Suitable ---
            # This is beneficial for columns with a limited number of unique string values.
            elif pd.api.types.is_object_dtype(optimized_df[col]):
                num_unique_values = optimized_df[col].nunique() # Count unique values in the column.
                num_total_values = len(optimized_df[col]) # Total number of values in the column.
                
                # Heuristic: Convert to 'category' if unique values are less than 50% of total
                # AND the number of unique values is less than 50. This prevents converting
                # columns with many unique strings (like IDs or free text) to category, which would be inefficient.
                if num_unique_values / num_total_values < 0.5 and num_unique_values < 50:
                    optimized_df[col] = optimized_df[col].astype('category')
                    self.logger.debug(f"Converted column {col} to category type. Unique values: {num_unique_values}")
        return optimized_df