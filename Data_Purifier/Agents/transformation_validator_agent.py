# agents/transformation_validator_agent.py
# This agent is responsible for validating the output of the TransformerAgent.
# It checks if data transformation operations like text processing and categorical
# encoding have been applied correctly.

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from data_purifier.utils.cached_chat_openai import CachedChatOpenAI as ChatOpenAI # For integrating with OpenAI LLMs
from crewai import Agent # Used for defining the agent's role, goal, and backstory


class TransformationValidatorAgent:
    """
    Agent for validating the transformation operations performed on a DataFrame.
    It ensures that data reshaping, text processing, and encoding operations
    yield the expected results.
    """

    def __init__(self, config: Dict = None, llm=None):
        """
        Initializes the TransformationValidatorAgent.

        Args:
            config (Dict, optional): Configuration dictionary for the agent.
            llm (optional): Language Model instance to be used by the agent.
                            Defaults to ChatOpenAI with gpt-4 if not provided.
        """
        # Initialize the LLM for the agent's reasoning capabilities
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory
        self.agent = Agent(
            role="Principal Data Transformation Quality Assurance Lead",
            goal="To conduct rigorous, comprehensive validation of all data transformation processes, ensuring that the reshaped, integrated, and encoded data assets are of the highest quality, perfectly aligned with analytical requirements, and optimized for downstream consumption, operating with the authority and precision of a lead data quality architect.",
            backstory="A highly experienced and detail-oriented data quality assurance lead with a deep specialization in validating complex data transformation pipelines. Possesses a comprehensive understanding of data integration, schema evolution, and the nuances of various encoding and text processing techniques. Excels at identifying subtle data discrepancies post-transformation and providing strategic recommendations to guarantee data integrity and fitness for purpose across all analytical and operational systems.",
            llm=self.llm,
            allow_delegation=False, # This agent does not delegate tasks to others
            verbose=True # Enable verbose output for the agent's actions
        )
        
        # Logger for internal logging within the agent
        self.logger = logging.getLogger(__name__)

    def validate_transformation(self, df: pd.DataFrame, transformation_instructions: Dict) -> Tuple[Dict, bool, Dict]:
        """
        Validates the transformation operations performed on the DataFrame.
        Checks for successful pivoting/unpivoting, merging/joining, text processing,
        and categorical encoding.

        Args:
            df (pd.DataFrame): The DataFrame after transformation operations have been applied.
            transformation_instructions (Dict): The instructions that were given to the TransformerAgent.

        Returns:
            Tuple[Dict, bool, Dict]: 
                - validation_report (Dict): A dictionary detailing the validation outcome and issues found.
                - validation_success (bool): True if validation passed, False otherwise.
                - recommendations (Dict): A dictionary of suggested transformation operations if issues are found.
        """
        validation_report = {
            "status": "success",
            "issues_found": {}, # Dictionary to store details of any issues identified
            "summary": "Transformation validation passed." # Overall summary message
        }
        validation_success = True # Flag to indicate overall validation success
        recommendations = {"transformation_operations": {}} # Dictionary to store recommendations for re-transformation

        self.logger.info("Starting transformation validation...")

        # --- 1. Validate Pivoting/Unpivoting ---
        # These operations can drastically change DataFrame shape and structure.
        # Generic validation is difficult without knowing the exact expected output shape.
        pivot_unpivot_instructions = transformation_instructions.get('pivoting_unpivoting', [])
        if pivot_unpivot_instructions:
            self.logger.info("Pivoting/Unpivoting instructions were present. Manual verification of results may be needed due to the complexity of generic validation.")
            # No automatic recommendations are generated for these due to their highly specific nature
            # and potential for data loss or unexpected reshaping if applied incorrectly.

        # --- 2. Validate Merging/Joining ---
        # Similar to pivoting, generic validation is complex as it depends on external data.
        merge_join_instructions = transformation_instructions.get('merging_joining', [])
        if merge_join_instructions:
            self.logger.info("Merging/Joining instructions were present. Manual verification of results may be needed due to dependency on external data and complex join conditions.")
            # No automatic recommendations for merge/join due to complexity and external data dependency

        # --- 3. Validate Text Processing ---
        # Checks if new text-processed columns exist and have the correct data type and content.
        text_processing_instructions = transformation_instructions.get('text_processing', {})
        for col_name, ops in text_processing_instructions.items():
            for op in ops:
                if op.get('operation') == 'text_process':
                    method = op.get('method')
                    new_column_name = op.get('new_column_name', f'{col_name}_{method}')

                    # Check 1: Was the new text-processed column created?
                    if new_column_name not in df.columns:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["text_processing_missing"] = f"Expected text processed feature '{new_column_name}' not found."
                        validation_report["summary"] = "Transformation validation failed: Text processing issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Expected text processed feature '{new_column_name}' not found.")
                        if col_name not in recommendations["transformation_operations"]:
                            recommendations["transformation_operations"][col_name] = []
                        recommendations["transformation_operations"][col_name].append({
                            "operation": "text_processing",
                            "method": method,
                            "reason": f"Text processed column '{new_column_name}' not found. Re-check text processing logic."
                        })
                    # Check 2: Is the new column of string type?
                    elif not pd.api.types.is_string_dtype(df[new_column_name]):
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["text_processing_type"] = f"Text processed feature '{new_column_name}' is not string type."
                        validation_report["summary"] = "Transformation validation failed: Text processing issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Text processed feature '{new_column_name}' is not string type.")
                        if col_name not in recommendations["transformation_operations"]:
                            recommendations["transformation_operations"][col_name] = []
                        recommendations["transformation_operations"][col_name].append({
                            "operation": "text_processing",
                            "method": method,
                            "reason": f"Text processed column '{new_column_name}' is not string type. Re-check text processing logic."
                        })

                    # Basic check for 'lowercase' transformation: ensure all characters are lowercase
                    if method == 'lowercase':
                        # Check if any non-empty string in the column contains uppercase characters
                        if not df[new_column_name].astype(str).apply(lambda x: x.islower() or not x.strip()).all():
                            validation_report["status"] = "failed"
                            validation_report["issues_found"]["text_processing_lowercase"] = f"Column '{new_column_name}' not fully lowercased."
                            validation_report["summary"] = "Transformation validation failed: Text processing issue."
                            validation_success = False
                            self.logger.warning(f"Validation Issue: Column '{new_column_name}' not fully lowercased.")
                            if col_name not in recommendations["transformation_operations"]:
                                recommendations["transformation_operations"][col_name] = []
                            recommendations["transformation_operations"][col_name].append({
                                "operation": "text_processing",
                                "method": "lowercase",
                                "reason": f"Column '{new_column_name}' not fully lowercased. Re-run lowercase transformation."
                            })

        # --- 4. Validate Categorical Encoding ---
        # Checks if encoding created new columns correctly and if original columns were handled.
        categorical_encoding_instructions = transformation_instructions.get('categorical_encoding', {})
        for col_name, ops in categorical_encoding_instructions.items():
            for op in ops:
                if op.get('operation') == 'encode_categorical':
                    method = op.get('method')

                    if method == 'one_hot':
                        # Check 1: For one-hot encoding, the original column is typically dropped.
                        if col_name in df.columns:
                            validation_report["status"] = "failed"
                            validation_report["issues_found"]["one_hot_original_not_dropped"] = f"Original column '{col_name}' not dropped after one-hot encoding."
                            validation_report["summary"] = "Transformation validation failed: Categorical encoding issue."
                            validation_success = False
                            self.logger.warning(f"Validation Issue: Original column '{col_name}' not dropped after one-hot encoding.")
                            if col_name not in recommendations["transformation_operations"]:
                                recommendations["transformation_operations"][col_name] = []
                            recommendations["transformation_operations"][col_name].append({
                                "operation": "encode_categorical",
                                "method": "one_hot",
                                "reason": f"Original column '{col_name}' not dropped. Re-run one-hot encoding."
                            })

                        # Check 2: Verify that new dummy columns were created and are numeric/boolean.
                        # This is a heuristic: check for columns with the expected prefix and numeric/bool type.
                        dummy_cols_found = [c for c in df.columns if c.startswith(f'{col_name}_') and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
                        if not dummy_cols_found:
                            validation_report["status"] = "failed"
                            validation_report["issues_found"]["one_hot_dummies_missing"] = f"No dummy columns found for '{col_name}' after one-hot encoding."
                            validation_report["summary"] = "Transformation validation failed: Categorical encoding issue."
                            validation_success = False
                            self.logger.warning(f"Validation Issue: No dummy columns found for '{col_name}' after one-hot encoding.")
                            if col_name not in recommendations["transformation_operations"]:
                                recommendations["transformation_operations"][col_name] = []
                            recommendations["transformation_operations"][col_name].append({
                                "operation": "encode_categorical",
                                "method": "one_hot",
                                "reason": f"Dummy columns for '{col_name}' not found. Re-run one-hot encoding."
                            })

                    elif method == 'label_encode' or method == 'frequency_encode':
                        new_column_name = f'{col_name}_encoded' if method == 'label_encode' else f'{col_name}_freq_encoded'
                        
                        # Check 1: Was the new encoded column created?
                        if new_column_name not in df.columns:
                            validation_report["status"] = "failed"
                            validation_report["issues_found"]["encoded_column_missing"] = f"Expected encoded column '{new_column_name}' not found."
                            validation_report["summary"] = "Transformation validation failed: Categorical encoding issue."
                            validation_success = False
                            self.logger.warning(f"Validation Issue: Expected encoded column '{new_column_name}' not found.")
                            if col_name not in recommendations["transformation_operations"]:
                                recommendations["transformation_operations"][col_name] = []
                            recommendations["transformation_operations"][col_name].append({
                                "operation": "encode_categorical",
                                "method": method,
                                "reason": f"Encoded column '{new_column_name}' not found. Re-run encoding."
                            })
                        # Check 2: Is the new encoded column numeric?
                        elif not pd.api.types.is_numeric_dtype(df[new_column_name]):
                            validation_report["status"] = "failed"
                            validation_report["issues_found"]["encoded_column_type"] = f"Encoded column '{new_column_name}' is not numeric type."
                            validation_report["summary"] = "Transformation validation failed: Categorical encoding issue."
                            validation_success = False
                            self.logger.warning(f"Validation Issue: Encoded column '{new_column_name}' is not numeric type.")
                            if col_name not in recommendations["transformation_operations"]:
                                recommendations["transformation_operations"][col_name] = []
                            recommendations["transformation_operations"][col_name].append({
                                "operation": "encode_categorical",
                                "method": method,
                                "reason": f"Encoded column '{new_column_name}' is not numeric. Re-run encoding."
                            })

        self.logger.info(f"Transformation validation finished with status: {validation_report['status']}")
        return validation_report, validation_success, recommendations