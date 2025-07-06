# agents/modifier_agent.py
# This agent is responsible for modifying the dataset by performing feature engineering,
# data aggregation, and other data restructuring operations.

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from crewai import Agent # Used for defining the agent's role, goal, and backstory
from langchain_openai import ChatOpenAI # For integrating with OpenAI LLMs


class DataModifierAgent:
    """
    Agent for performing data modification operations such as feature engineering,
    aggregation, column renaming/dropping, and discretization/binning.
    """

    def __init__(self, config: Dict = None, llm=None):
        """
        Initializes the DataModifierAgent.

        Args:
            config (Dict, optional): Configuration dictionary for the agent.
            llm (optional): Language Model instance to be used by the agent.
                            Defaults to ChatOpenAI with gpt-4 if not provided.
        """
        # Initialize the LLM for the agent's reasoning capabilities
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory
        self.agent = Agent(
            role="Principal Feature Engineer & Data Modeler",
            goal="To expertly design and implement advanced feature engineering techniques, data aggregation strategies, and data restructuring operations to create optimal datasets for analytical and machine learning applications, mimicking the strategic approach of a lead data scientist.",
            backstory="A highly innovative and experienced data scientist specializing in the art and science of data modification. Possesses a deep understanding of how to transform raw data into high-value analytical assets through sophisticated feature creation, intelligent aggregation, and precise data modeling, ensuring the data is perfectly tailored for complex analytical tasks and predictive models.",
            llm=self.llm,
            allow_delegation=False, # This agent does not delegate tasks to others
            verbose=True # Enable verbose output for the agent's actions
        )
        
        # List to store logs of modification operations
        self.modification_logs = []
        # Logger for internal logging within the agent
        self.logger = logging.getLogger(__name__)

    def _log_modification(self, message: str, reason: str = None):
        """
        Logs a modification operation message with a timestamp and an optional reason.
        This is for internal tracking and console output.

        Args:
            message (str): The message describing the modification operation.
            reason (str, optional): The reason behind the operation.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "message": message, "reason": reason}
        self.modification_logs.append(log_entry)
        print(f"[Modification] {message}" + (f" (Reason: {reason})" if reason else ""))

    def modify_dataset(self, df: pd.DataFrame, processing_instructions: Dict, learned_optimizations: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Processes the input DataFrame according to the modification instructions.
        This is the main method for the DataModifierAgent.

        Args:
            df (pd.DataFrame): The input DataFrame to be modified.
            processing_instructions (Dict): A dictionary of modification operations to apply.
                                            Typically derived from MetaAnalyzerAgent's suggestions.
            learned_optimizations (Optional[Dict]): Optimizations learned from previous validation failures.

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing the modified DataFrame and a dictionary
                                       of modification reports.
        """
        modification_report = {
            "operations_performed": [] # List to store details of each operation performed
        }
        modified_df = df.copy() # Work on a copy to avoid modifying the original DataFrame directly

        # Get modification-specific operations from the overall processing instructions
        mod_ops = processing_instructions.get('modification_operations', {})

        # Iterate through the modification instructions provided
        # Instructions are typically structured as {column_name: [list_of_operations]}
        for target_key, operations_list in mod_ops.items():
            # Ensure the operations list is indeed a list
            if not isinstance(operations_list, list):
                self.logger.warning(f"Expected a list of operations for {target_key}, but got {type(operations_list)}. Skipping.")
                continue

            # Apply each operation for the current target (column or global)
            for op_details in operations_list:
                operation_type = op_details.get('operation')
                status = "failed" # Default status
                details = {} # Details of the operation outcome
                column_affected = target_key # Default to target_key as the affected column

                try:
                    # Handle feature engineering operations
                    if operation_type == 'create_new_feature':
                        modified_df, status, details = self._apply_feature_engineering(modified_df, op_details)
                        column_affected = op_details.get('new_feature_name', target_key) # New feature name is the affected column

                    # Handle data aggregation operations
                    elif operation_type == 'aggregate':
                        modified_df, status, details = self._apply_single_aggregation(modified_df, op_details)
                        column_affected = op_details.get('target_column', target_key) # Target column of aggregation

                    # Handle column renaming or dropping operations
                    elif operation_type == 'rename_column' or operation_type == 'drop_column':
                        modified_df, status, details = self._apply_column_operation(modified_df, op_details)
                        column_affected = op_details.get('old_name', op_details.get('column_name', target_key)) # Original column name

                    # Handle discretization/binning operations
                    elif operation_type == 'discretize_bin':
                        modified_df, status, details = self._apply_single_discretization_binning(modified_df, target_key, op_details)
                        column_affected = op_details.get('new_column_name', f'{target_key}_binned') # New binned column name

                    # Log unknown operations
                    else:
                        self.logger.warning(f"Unknown modification operation type: {operation_type} for {target_key}. Skipping.")
                        status = "skipped"
                        details = f"Unknown operation type: {operation_type}"

                except Exception as e:
                    # Catch any exceptions during operation application and log them
                    error_message = f"Error applying modification operation {operation_type} on {column_affected}: {e}"
                    self.logger.error(error_message, exc_info=True)
                    status = "failed"
                    details = str(e)

                # Record details of the performed operation
                modification_report["operations_performed"].append({
                    "operation": operation_type,
                    "column": column_affected,
                    "status": status,
                    "details": details
                })

        return modified_df, modification_report

    def _apply_column_operation(self, df: pd.DataFrame, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Handles a single column renaming or dropping operation.

        Args:
            df (pd.DataFrame): The input DataFrame.
            op_config (Dict): Configuration for the column operation.

        Returns:
            Tuple[pd.DataFrame, str, Dict]: The modified DataFrame, status, and details.
        """
        status = "failed"
        details = {}
        operation_type = op_config.get('operation')

        try:
            if operation_type == 'rename_column':
                old_name = op_config.get('old_name')
                new_name = op_config.get('new_name')
                if old_name in df.columns: # Check if the column exists
                    df = df.rename(columns={old_name: new_name}) # Rename the column
                    self._log_modification(f"Renamed column '{old_name}' to '{new_name}'.", reason=f"Column renamed from {old_name} to {new_name} as per instructions.")
                    status = "completed"
                    details = {"old_name": old_name, "new_name": new_name}
                else:
                    self._log_modification(f"Rename failed: Column '{old_name}' not found.", reason=f"Skipped renaming as column {old_name} was not found.")
                    status = "skipped"
                    details = {"error": f"Column {old_name} not found"}
            elif operation_type == 'drop_column':
                column_to_drop = op_config.get('column_name')
                if column_to_drop in df.columns: # Check if the column exists
                    df = df.drop(columns=[column_to_drop]) # Drop the column
                    self._log_modification(f"Dropped column '{column_to_drop}'.", reason=f"Column {column_to_drop} dropped as per instructions.")
                    status = "completed"
                    details = {"column_name": column_to_drop}
                else:
                    self._log_modification(f"Drop failed: Column '{column_to_drop}' not found.", reason=f"Skipped dropping as column {column_to_drop} was not found.")
                    status = "skipped"
                    details = {"error": f"Column {column_to_drop} not found"}
            else:
                status = "skipped"
                details = {"error": f"Unknown column operation type: {operation_type}"}
        except Exception as e:
            self._log_modification(f"Error in column operation {operation_type}: {e}", reason=f"Error during column operation: {str(e)}")
            status = "failed"
            details = {"error": str(e)}
        return df, status, details

    def _apply_feature_engineering(self, df: pd.DataFrame, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Handles a single feature engineering operation, typically involving creating a new column
        based on a provided formula.

        Args:
            df (pd.DataFrame): The input DataFrame.
            op_config (Dict): Configuration for the feature engineering operation.

        Returns:
            Tuple[pd.DataFrame, str, Dict]: The modified DataFrame, status, and details.
        """
        status = "failed"
        details = {}
        operation_type = op_config.get('operation')

        try:
            if operation_type == 'create_new_feature':
                formula = op_config.get('formula') # The formula to evaluate (e.g., 'df["col1"] + df["col2"]')
                new_feature_name = op_config.get('new_feature_name', 'engineered_feature')
                method = op_config.get('method') # New: method for adaptive feature engineering

                if formula: # If a specific formula is provided, use it
                    try:
                        # Evaluate the formula using eval(). 'df' is provided in the local scope.
                        df[new_feature_name] = eval(formula, {'df': df})
                        self._log_modification(f"Created new feature '{new_feature_name}' using formula: {formula}", reason=f"New feature {new_feature_name} created based on formula: {formula}")
                        status = "completed"
                        details = {"formula": formula, "new_feature_name": new_feature_name}
                    except Exception as e:
                        self._log_modification(f"Failed to create feature '{new_feature_name}': {e}", reason=f"Failed to create feature due to error: {str(e)}")
                        status = "failed"
                        details = {"error": str(e)}
                elif method == 'age_grouping': # Adaptive: age grouping
                    if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
                        bins = [0, 18, 35, 55, 100]
                        labels = ['0-18', '19-35', '36-55', '56+']
                        df[new_feature_name] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
                        self._log_modification(f"Created age groups in '{new_feature_name}'.", reason="Adaptive: Age grouping applied to create categorical age ranges.")
                        status = "completed"
                        details = {"method": method, "new_feature_name": new_feature_name}
                    else:
                        status = "skipped"
                        details = "Age column not found or not numeric for age grouping."
                elif method == 'bmi_categories': # Adaptive: BMI categorization
                    if 'bmi' in df.columns and pd.api.types.is_numeric_dtype(df['bmi']):
                        bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, 100]
                        labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity Class I', 'Obesity Class II', 'Obesity Class III']
                        df[new_feature_name] = pd.cut(df['bmi'], bins=bins, labels=labels, right=True)
                        self._log_modification(f"Created BMI categories in '{new_feature_name}'.", reason="Adaptive: BMI categorization applied to create health-related categories.")
                        status = "completed"
                        details = {"method": method, "new_feature_name": new_feature_name}
                    else:
                        status = "skipped"
                        details = "BMI column not found or not numeric for BMI categorization."
                else:
                    status = "skipped"
                    details = {"error": f"Unknown or unsupported feature engineering method: {method}"}
            else:
                status = "skipped"
                details = {"error": f"Unknown feature engineering operation type: {operation_type}"}
        except Exception as e:
            self._log_modification(f"Error in feature engineering operation {operation_type}: {e}", reason=f"Error during feature engineering: {str(e)}")
            status = "failed"
            details = {"error": str(e)}
        return df, status, details

    def _apply_single_aggregation(self, df: pd.DataFrame, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Handles a single data aggregation operation.

        Args:
            df (pd.DataFrame): The input DataFrame.
            op_config (Dict): Configuration for the aggregation operation.

        Returns:
            Tuple[pd.DataFrame, str, Dict]: The aggregated DataFrame, status, and details.
        """
        status = "failed"
        details = {}
        group_by_cols_str = op_config.get('group_by_cols') # Comma-separated string of columns to group by
        target_column = op_config.get('target_column') # Column to aggregate
        agg_method = op_config.get('method') # Aggregation method (e.g., 'sum', 'mean', 'count')

        try:
            # Validate required parameters
            if not group_by_cols_str or not target_column or not agg_method:
                details = {"error": "Missing required parameters for aggregation (group_by_cols, target_column, method)."}
                return df, status, details

            group_by_cols = [col.strip() for col in group_by_cols_str.split(',')] # Parse group-by columns
            if not all(col in df.columns for col in group_by_cols): # Check if group-by columns exist
                details = {"error": f"Group-by columns not found: {group_by_cols}"}
                return df, status, details

            if target_column not in df.columns: # Check if target column exists
                details = {"error": f"Target column not found: {target_column}"}
                return df, status, details

            # Perform the aggregation
            aggregated_df = df.groupby(group_by_cols)[target_column].agg(agg_method).reset_index()
            self._log_modification(f"Aggregated data by {group_by_cols} on {target_column} using {agg_method}.", reason=f"Data aggregated to summarize {target_column} by {group_by_cols}.")
            status = "completed"
            details = {"group_by": group_by_cols, "target_column": target_column, "method": agg_method}
            return aggregated_df, status, details
        except Exception as e:
            self._log_modification(f"Aggregation failed for {target_column} with {agg_method}: {e}", reason=f"Aggregation failed due to error: {str(e)}")
            status = "failed"
            details = {"error": str(e)}
            return df, status, details

    def _apply_single_discretization_binning(self, df: pd.DataFrame, col_name: str, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Handles a single discretization (binning) operation for a numeric column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            col_name (str): The name of the column to discretize.
            op_config (Dict): Configuration for the binning operation.

        Returns:
            Tuple[pd.DataFrame, str, Dict]: The modified DataFrame, status, and details.
        """
        status = "failed"
        details = {}
        method = op_config.get('method') # Binning method ('equal_width' or 'equal_frequency')
        bins = op_config.get('bins') # Number of bins or list of bin edges
        labels = op_config.get('labels') # Optional labels for the bins
        new_column_name = op_config.get('new_column_name', f'{col_name}_binned') # Name for the new binned column

        try:
            # Validate column existence and type
            if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
                error_message = f"Column '{col_name}' not found or not numeric for binning."
                self.logger.error(error_message)
                details = {"error": error_message}
                return df, status, details

            if method == 'equal_width':
                df[new_column_name] = pd.cut(df[col_name], bins=bins, labels=labels)
            elif method == 'equal_frequency':
                df[new_column_name] = pd.qcut(df[col_name], q=bins, labels=labels, duplicates='drop')
            else:
                error_message = f"Unknown binning method: {method}"
                self.logger.error(error_message)
                details = {"error": error_message}
                return df, status, details

            self._log_modification(f"Discretized/Binned column '{col_name}' into '{new_column_name}' using {method}.", reason=f"Column {col_name} binned into {new_column_name} for categorical analysis.")
            status = "completed"
            details = {"method": method, "bins": bins, "new_column_name": new_column_name}
            return df, status, details
        except Exception as e:
            self._log_modification(f"Binning failed for '{col_name}': {e}", reason=f"Binning failed due to error: {str(e)}")
            status = "failed"
            details = {"error": str(e)}
            return df, status, details