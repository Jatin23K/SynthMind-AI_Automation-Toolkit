# agents/cleaner_agent.py
# This agent is responsible for performing various data cleaning operations.
# It identifies and rectifies common data quality issues like missing values, outliers, duplicates, and inconsistencies.

import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import multiprocessing

import numpy as np
import pandas as pd
from crewai import Agent # Used for defining the agent's role, goal, and backstory
from fuzzywuzzy import fuzz # For fuzzy string matching in inconsistency handling
from langchain_openai import ChatOpenAI # For integrating with OpenAI LLMs
from sklearn.ensemble import IsolationForest # For outlier detection
from sklearn.impute import SimpleImputer # For handling missing values

from utils.report_generator import ReportGenerator # Utility for generating reports (though not directly used for final report here)

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

class CleanerAgent:
    """
    Agent for performing data cleaning operations.
    It encapsulates the logic for handling missing values, outliers, duplicates,
    and inconsistencies within a dataset.
    """

    def __init__(self, config: Dict = None, llm=None):
        """
        Initializes the CleanerAgent.

        Args:
            config (Dict, optional): Configuration dictionary for the agent.
            llm (optional): Language Model instance to be used by the agent.
                            Defaults to ChatOpenAI with gpt-4 if not provided.
        """
        # Initialize the LLM for the agent's reasoning capabilities
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory
        self.agent = Agent(
            role="Principal Data Quality Engineer",
            goal="To meticulously analyze raw datasets, proactively identify and rectify all data quality anomalies (missing values, outliers, duplicates, inconsistencies), and apply advanced cleaning strategies to ensure the dataset is impeccably clean, reliable, and ready for sophisticated analysis, mirroring the precision of a top-tier data quality expert.",
            backstory="A veteran data quality engineer with an unparalleled eye for detail and a profound understanding of data integrity principles. Possesses extensive experience in diagnosing complex data issues, implementing robust and adaptive cleaning algorithms, and ensuring data assets meet the highest standards of accuracy and consistency for critical business intelligence and machine learning initiatives.",
            llm=self.llm,
            allow_delegation=False, # This agent does not delegate tasks to others
            verbose=True # Enable verbose output for the agent's actions
        )
        
        # Logger for internal logging within the agent
        self.logger = logging.getLogger(__name__)
        
        # Lists and dictionaries to store cleaning process details and statistics
        self.cleaning_logs = [] # Stores a chronological log of cleaning operations
        self.cleaning_stats = {} # Stores summary statistics after cleaning
        self.cleaning_config = config or {
            "max_unique_for_fuzzy_matching": 1000 # Default threshold for fuzzy matching
        } # Configuration specific to cleaning operations
        self.imputers = {} # Stores imputer models (not fully utilized in parallel mode)
        self.outlier_detectors = {} # Stores outlier detector models
        self.operation_reports = {} # Stores reports for individual operations

        # ReportGenerator instance (currently not used for the final report, but can be extended)
        self.report_generator = ReportGenerator()
        self.cleaning_operations = [] # List to track planned cleaning operations

    def _log_cleaning(self, message: str, reason: str = None):
        """
        Logs a cleaning operation message with a timestamp and an optional reason.
        This is for internal tracking and console output.

        Args:
            message (str): The message describing the cleaning operation.
            reason (str, optional): The reason behind the operation.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "message": message, "reason": reason}
        self.cleaning_logs.append(log_entry)
        print(f"[Cleaning] {message}" + (f" (Reason: {reason})" if reason else ""))

    def _log_operation(self, operation: str, details: Dict):
        """
        Logs detailed information about a specific cleaning operation.

        Args:
            operation (str): The type of operation performed (e.g., 'handle_missing_values').
            details (Dict): A dictionary containing specific details and statistics of the operation.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'details': details
        }
        self.cleaning_logs.append(log_entry)
        print(f"[{timestamp}] {operation}: {json.dumps(details, indent=2)}")

    def _validate_operation_result(self, df: pd.DataFrame, operation: str) -> bool:
        """
        Placeholder for calling a dedicated validator agent.
        In this architecture, the actual validation is performed by CleaningValidatorAgent
        after the CleanerAgent completes its work.

        Args:
            df (pd.DataFrame): The DataFrame after an operation.
            operation (str): The operation that was performed.

        Returns:
            bool: Always True, as actual validation is external.
        """
        return True

    def _log_validation_issue(self, message: str):
        """
        Logs a warning message related to a validation issue.

        Args:
            message (str): The validation issue message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.warning(f"[{timestamp}] Validation Issue: {message}")

    def clean_dataset(self, df: pd.DataFrame, cleaning_instructions: Optional[Dict] = None, learned_optimizations: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Cleans the input DataFrame based on provided instructions and learned optimizations.
        This is the main method for the CleanerAgent.

        Args:
            df (pd.DataFrame): The input DataFrame to be cleaned.
            cleaning_instructions (Optional[Dict]): A dictionary of cleaning operations to apply.
                                                    Typically derived from MetaAnalyzerAgent's suggestions.
            learned_optimizations (Optional[Dict]): Optimizations learned from previous validation failures.

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing the cleaned DataFrame and a dictionary
                                       of cleaning statistics and reports.
        """
        # Update cleaning configuration with new instructions (e.g., from Orchestrator's retries)
        self.cleaning_config.update(cleaning_instructions or {})
        self.cleaning_stats = {} # Reset stats for current run
        self.operation_reports = {} # Reset operation reports

        cleaned_df = df.copy() # Work on a copy to avoid modifying the original DataFrame directly
        operations_performed_details = [] # To log details of each operation performed

        # Iterate through the cleaning instructions provided
        # Instructions are typically structured as {column_name: [list_of_operations]}
        for target, ops_list in cleaning_instructions.items():
            # Ensure the operations list is indeed a list
            if not isinstance(ops_list, list):
                self.logger.warning(f"Expected a list of operations for {target}, but got {type(ops_list)}. Skipping.")
                continue

            # Apply each operation for the current target (column or global)
            for op_details in ops_list:
                operation_type = op_details.get('operation')
                status = "failed" # Default status
                details = {} # Details of the operation outcome
                column_affected = target # Default to target as the affected column

                try:
                    # Handle missing values for a specific column
                    if operation_type == 'handle_missing_values':
                        cleaned_df = self._handle_missing_values(cleaned_df, cleaning_instructions)
                        status = "completed"
                        details = "Missing values handled in parallel across relevant columns."
                        column_affected = "All relevant columns"

                    # Handle outliers for a specific column
                    elif operation_type == 'handle_outliers':
                        col_name = target
                        if col_name in cleaned_df.columns:
                            cleaned_df[col_name], status, details = self._apply_outlier_handling(cleaned_df[col_name], op_details.get('method'))
                            column_affected = col_name
                        else:
                            status = "skipped"
                            details = f"Column {col_name} not found for outlier handling."

                    # Remove duplicate rows (can be global or subset-based)
                    elif operation_type == 'remove_duplicates':
                        subset = op_details.get('subset') # Columns to consider for duplicates
                        original_len = len(cleaned_df)
                        # If target is 'global' or subset is None, apply to entire DataFrame
                        if target == 'global' or subset is None:
                            cleaned_df = self.remove_duplicates(cleaned_df, subset=subset)
                            removed_count = original_len - len(cleaned_df)
                            status = "completed"
                            details = f"Removed {removed_count} duplicate rows, subset: {subset}"
                        else:
                            # If a specific column is targeted, but it's a duplicate removal, it's likely a misconfiguration
                            status = "skipped"
                            details = f"Duplicate removal operation for specific column {target} is not supported. Use 'global' target."

                    # Handle inconsistencies for a specific column using fuzzy matching
                    elif operation_type == 'handle_inconsistencies':
                        col_name = target
                        # Check if fuzzy matching is enabled in the cleaning config
                        if self.cleaning_config.get('enable_fuzzy_matching', True):
                            if col_name in cleaned_df.columns:
                                cleaned_df[col_name], status, details = self._apply_inconsistency_handling(cleaned_df[col_name], op_details.get('method'))
                                column_affected = col_name
                            else:
                                status = "skipped"
                                details = f"Column {col_name} not found for inconsistency handling."
                        else:
                            status = "skipped"
                            details = "Fuzzy matching disabled."

                    # Log unknown operations
                    else:
                        status = "skipped"
                        details = f"Unknown cleaning operation type: {operation_type}"

                except Exception as e:
                    # Catch any exceptions during operation application and log them
                    error_message = f"Error applying cleaning operation {operation_type} on {column_affected}: {e}"
                    self.logger.error(error_message, exc_info=True)
                    status = "failed"
                    details = str(e)

                # Record details of the performed operation
                operations_performed_details.append({
                    "operation": operation_type,
                    "column": column_affected,
                    "status": status,
                    "details": details
                })

        # Generate final cleaning statistics after all operations are attempted
        self._generate_cleaning_stats(cleaned_df)

        # Generate a comprehensive cleaning report (for internal use/logging)
        final_report = self.generate_cleaning_report()
        self.cleaning_stats['report'] = final_report
        self.cleaning_stats['operation_reports'] = self.operation_reports
        self.cleaning_stats['operations_performed_details'] = operations_performed_details

        return cleaned_df, self.cleaning_stats

    def _apply_outlier_handling(self, series: pd.Series, method: str) -> Tuple[pd.Series, str, str]:
        """
        Applies outlier handling to a single Pandas Series (column).

        Args:
            series (pd.Series): The input Series.
            method (str): The outlier handling method to use (e.g., 'isolation_forest').

        Returns:
            Tuple[pd.Series, str, str]: The Series with outliers handled, status, and details.
        """
        series = series.copy() # Work on a copy
        status = "failed"
        details = ""
        try:
            if pd.api.types.is_numeric_dtype(series): # Outlier handling typically applies to numeric data
                method_used = method # Track the method actually used
                if method == 'isolation_forest':
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(series.to_frame())
                    original_outlier_count = (outliers == -1).sum()
                    if original_outlier_count > 0:
                        series.loc[outliers == -1] = series.median()
                        status = "completed"
                        details = f"Outliers handled using {method}. Replaced {original_outlier_count} outliers with median."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method}."
                        self.logger.info(details)
                elif method == 'iqr_clipping':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    original_outlier_count = series[(series < lower_bound) | (series > upper_bound)].count()
                    if original_outlier_count > 0:
                        series = series.clip(lower=lower_bound, upper=upper_bound)
                        status = "completed"
                        details = f"Outliers handled using {method}. Clipped {original_outlier_count} outliers."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method}."
                        self.logger.info(details)
                elif method == 'z_score_removal':
                    from scipy.stats import zscore
                    z_scores = np.abs(zscore(series.dropna()))
                    original_outlier_count = series[z_scores > 3].count() # Z-score threshold of 3
                    if original_outlier_count > 0:
                        series = series[z_scores <= 3] # Remove outliers
                        status = "completed"
                        details = f"Outliers handled using {method}. Removed {original_outlier_count} outliers."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method}."
                        self.logger.info(details)
                else: # Adaptive choice for numeric if no specific method or unsupported
                    # Default to IQR clipping for robustness if no specific method is provided or supported
                    method_used = 'iqr_clipping (adaptive)'
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    original_outlier_count = series[(series < lower_bound) | (series > upper_bound)].count()
                    if original_outlier_count > 0:
                        series = series.clip(lower=lower_bound, upper=upper_bound)
                        status = "completed"
                        details = f"Outliers handled using {method_used}. Clipped {original_outlier_count} outliers."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method_used}."
                        self.logger.info(details)
            else:
                status = "skipped"
                details = "Column is not numeric for outlier handling. Skipping."
                self.logger.warning(details)
        except Exception as e:
            status = "failed"
            details = f"Error during outlier handling with method {method}: {str(e)}"
            self.logger.error(details, exc_info=True)
        return series, status, details

    def _apply_missing_value_imputation(self, series: pd.Series, method: str) -> Tuple[pd.Series, str, str]:
        """
        Applies missing value imputation to a single Pandas Series (column).

        Args:
            series (pd.Series): The input Series.
            method (str): The imputation method to use ('mean', 'median', 'mode').

        Returns:
            Tuple[pd.Series, str, str]: The Series with missing values imputed, status, and details.
        """
        status = "failed"
        details = ""
        try:
            if series.isnull().any(): # Only proceed if there are missing values
                imputer = None
                if pd.api.types.is_numeric_dtype(series):
                    if method == 'mean' or method == 'mean_imputation':
                        imputer = SimpleImputer(strategy='mean')
                        method_used = 'mean'
                    elif method == 'median':
                        imputer = SimpleImputer(strategy='median')
                        method_used = 'median'
                    else: # Adaptive choice for numeric if no specific method or unsupported
                        if series.skew() > 1 or series.skew() < -1: # Check for high skewness
                            imputer = SimpleImputer(strategy='median')
                            method_used = 'median (adaptive)'
                        else:
                            imputer = SimpleImputer(strategy='mean')
                            method_used = 'mean (adaptive)'
                else: # For non-numeric (categorical/object) columns
                    if method == 'mode' or method == 'mode_imputation':
                        imputer = SimpleImputer(strategy='most_frequent')
                        method_used = 'mode'
                    else: # Adaptive choice for non-numeric if no specific method or unsupported
                        imputer = SimpleImputer(strategy='most_frequent')
                        method_used = 'mode (adaptive)'

                if imputer:
                    # Apply imputation and convert back to Series
                    imputed_series = pd.Series(imputer.fit_transform(series.to_frame())[:, 0], index=series.index, name=series.name)
                    status = "completed"
                    details = f"Missing values handled using {method} strategy."
                    return imputed_series, status, details
                else:
                    details = f"Unknown or unsupported imputation method: {method}."
            else:
                status = "completed"
                details = "No missing values to handle." # No action needed if no NaNs
        except Exception as e:
            details = str(e)
        return series, status, details

    def _apply_inconsistency_handling(self, series: pd.Series, method: str) -> Tuple[pd.Series, str, str]:
        """
        Applies inconsistency handling to a single Pandas Series, primarily for categorical data.
        Uses fuzzy matching to identify and map similar-looking values to a canonical form.

        Args:
            series (pd.Series): The input Series.

        Returns:
            Tuple[pd.Series, str, str]: The Series with inconsistencies handled, status, and details.
        """
        status = "failed"
        details = ""
        method_used = method # To track the method actually used
        try:
            if self._is_categorical_column(series): # Only apply to columns identified as categorical
                max_unique = self.cleaning_config.get('max_unique_for_fuzzy_matching', 1000)

                if method == 'standardize_case':
                    series = series.astype(str).str.strip().str.lower()
                    status = "completed"
                    details = "Inconsistencies handled by standardizing case and trimming whitespace."
                elif method == 'fuzzy_match':
                    if series.nunique() > max_unique:
                        status = "skipped"
                        details = f"Skipping fuzzy matching for column '{series.name}' due to high cardinality ({series.nunique()} unique values > {max_unique})."
                        self.logger.warning(details)
                        return series, status, details

                    value_counts = series.value_counts()
                    unique_values_sorted = value_counts.sort_values(ascending=False).index.tolist()

                    mapping = {}
                    processed_values = set()

                    for canonical_val in unique_values_sorted:
                        if canonical_val in processed_values:
                            continue

                        processed_values.add(canonical_val)

                        for other_val in unique_values_sorted:
                            if other_val == canonical_val or other_val in processed_values:
                                continue

                            ratio = fuzz.ratio(str(canonical_val), str(other_val))
                            partial_ratio = fuzz.partial_ratio(str(canonical_val), str(other_val))
                            token_sort_ratio = fuzz.token_sort_ratio(str(canonical_val), str(other_val))

                            if max(ratio, partial_ratio, token_sort_ratio) > 80:
                                if value_counts[canonical_val] >= value_counts[other_val]:
                                    mapping[other_val] = canonical_val
                                else:
                                    mapping[canonical_val] = other_val
                                processed_values.add(other_val)

                    if mapping:
                        series = series.replace(mapping)
                        status = "completed"
                        details = f"Handled inconsistencies by fuzzy matching and mapping {len(mapping)} values."
                    else:
                        status = "completed"
                        details = "No inconsistencies found or mapped via fuzzy matching."
                else: # Adaptive choice if no specific method or unsupported
                    method_used = 'adaptive_standardize_and_fuzzy'
                    # First, standardize case and trim whitespace
                    original_series = series.copy()
                    series = series.astype(str).str.strip().str.lower()

                    if series.nunique() > max_unique:
                        status = "skipped"
                        details = f"Skipping adaptive fuzzy matching for column '{series.name}' due to high cardinality ({series.nunique()} unique values > {max_unique})."
                        self.logger.warning(details)
                        return original_series, status, details # Return original if skipped

                    value_counts = series.value_counts()
                    unique_values_sorted = value_counts.sort_values(ascending=False).index.tolist()

                    mapping = {}
                    processed_values = set()

                    for canonical_val in unique_values_sorted:
                        if canonical_val in processed_values:
                            continue

                        processed_values.add(canonical_val)

                        for other_val in unique_values_sorted:
                            if other_val == canonical_val or other_val in processed_values:
                                continue

                            ratio = fuzz.ratio(str(canonical_val), str(other_val))
                            partial_ratio = fuzz.partial_ratio(str(canonical_val), str(other_val))
                            token_sort_ratio = fuzz.token_sort_ratio(str(canonical_val), str(other_val))

                            if max(ratio, partial_ratio, token_sort_ratio) > 80:
                                if value_counts[canonical_val] >= value_counts[other_val]:
                                    mapping[other_val] = canonical_val
                                else:
                                    mapping[canonical_val] = other_val
                                processed_values.add(other_val)

                    if mapping:
                        # Apply mapping to the original series, then re-apply standardized case/trim
                        series = original_series.replace(mapping)
                        series = series.astype(str).str.strip().str.lower()
                        status = "completed"
                        details = f"Handled inconsistencies by adaptive fuzzy matching and mapping {len(mapping)} values."
                    else:
                        status = "completed"
                        details = "No inconsistencies found or mapped via adaptive fuzzy matching."
            else:
                status = "completed"
                details = "Column is not categorical for inconsistency handling." # Skip if not categorical
        except Exception as e:
            details = str(e)
        return series, status, details

    

    def _analyze_outliers_iqr(self, series: pd.Series) -> Dict:
        """
        Analyzes outliers in a numeric series using the Interquartile Range (IQR) method.

        Args:
            series (pd.Series): The input numeric Series.

        Returns:
            Dict: A dictionary containing outlier statistics (count, ratio).
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        stats = {
            'outliers': {
                'count': len(outliers),
                'ratio': len(outliers) / len(series.dropna()) if len(series.dropna()) > 0 else 0
            }
        }
        return stats

    @staticmethod
    def _impute_column_parallel(args):
        """
        Static method to perform missing value imputation on a single column.
        Designed to be used with multiprocessing for parallel execution.

        Args:
            args (Tuple): A tuple containing (column_name, series, strategy, outlier_stats).

        Returns:
            Tuple: (column_name, imputed_series, log_message).
        """
        column_name, series, strategy, outlier_stats = args
        imputer = None
        log_message = ""

        if pd.api.types.is_numeric_dtype(series):
            # If strategy is not specified, dynamically choose based on outlier presence
            if strategy not in ['mean', 'median']:
                if outlier_stats and outlier_stats['outliers']['ratio'] > 0.05:
                    strategy = 'median' # Use median if significant outliers are present
                    log_message = f"Using median imputation for '{column_name}' due to {outlier_stats['outliers']['ratio']:.2%} outliers"
                else:
                    strategy = 'mean' # Use mean if no significant outliers
                    log_message = f"Using mean imputation for '{column_name}' (no significant outliers)"
            
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
        else:
            # For non-numeric columns, default to mode imputation
            strategy = 'mode'
            imputer = SimpleImputer(strategy='most_frequent')
            log_message = f"Column '{column_name}' is non-numeric, using {strategy} imputation"

        if imputer:
            # Apply imputation and return the imputed series
            imputed_series = pd.Series(imputer.fit_transform(series.to_frame())[:, 0], index=series.index, name=column_name)
            log_message += f"\nHandled missing values in '{column_name}' using {strategy} strategy"
        else:
            imputed_series = series.copy() # If no imputer, return original series
            log_message = f"No imputation performed for '{column_name}'"

        return column_name, imputed_series, log_message

    def _handle_missing_values(self, df: pd.DataFrame, cleaning_instructions: Dict) -> pd.DataFrame:
        """
        Handles missing values across the DataFrame, potentially using parallel processing.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cleaning_instructions (Dict): The cleaning instructions containing per-column methods.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        columns_to_impute = []
        for column in df.columns:
            if df[column].isnull().any(): # Check for columns with missing values
                # Get the specific strategy for this column from cleaning_instructions
                column_ops = cleaning_instructions.get(column, [])
                strategy = None
                for op in column_ops:
                    if op.get('operation') == 'handle_missing_values':
                        strategy = op.get('method')
                        break

                outlier_stats = None
                if pd.api.types.is_numeric_dtype(df[column]):
                    outlier_stats = self._analyze_outliers_iqr(df[column]) # Analyze outliers for numeric columns
                columns_to_impute.append((column, df[column], strategy, outlier_stats))

        if not columns_to_impute:
            return df # Return original DataFrame if no columns need imputation

        # Determine the number of processes to use for parallel imputation
        num_processes = min(len(columns_to_impute), multiprocessing.cpu_count()) # Use min of columns or CPU count
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(CleanerAgent._impute_column_parallel, columns_to_impute) # Map imputation to pool

        imputed_df = df.copy() # Create a copy to update
        for col_name, imputed_series, log_message in results:
            imputed_df[col_name] = imputed_series # Update the column with imputed values
            self._log_cleaning(log_message, reason=f"Imputation performed using {method_used} strategy.") # Log the imputation
            self.imputers[col_name] = "Imputer details (not stored per-column in parallel mode)" # Placeholder for imputer details

        return imputed_df

    @staticmethod
    def _impute_column_parallel(args):
        """
        Static method to perform missing value imputation on a single column.
        Designed to be used with multiprocessing for parallel execution.

        Args:
            args (Tuple): A tuple containing (column_name, series, strategy, outlier_stats).

        Returns:
            Tuple: (column_name, imputed_series, log_message).
        """
        column_name, series, strategy, outlier_stats = args
        imputer = None
        log_message = ""

        if pd.api.types.is_numeric_dtype(series):
            # If strategy is not specified, dynamically choose based on outlier presence
            if strategy not in ['mean', 'median']:
                if outlier_stats and outlier_stats['outliers']['ratio'] > 0.05:
                    strategy = 'median' # Use median if significant outliers are present
                    log_message = f"Using median imputation for '{column_name}' due to {outlier_stats['outliers']['ratio']:.2%} outliers"
                else:
                    strategy = 'mean' # Use mean if no significant outliers
                    log_message = f"Using mean imputation for '{column_name}' (no significant outliers)"
            
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
        else:
            # For non-numeric columns, default to mode imputation
            strategy = 'mode'
            imputer = SimpleImputer(strategy='most_frequent')
            log_message = f"Column '{column_name}' is non-numeric, using {strategy} imputation"

        if imputer:
            # Apply imputation and return the imputed series
            imputed_series = pd.Series(imputer.fit_transform(series.to_frame())[:, 0], index=series.index, name=column_name)
            log_message += f"\nHandled missing values in '{column_name}' using {strategy} strategy"
        else:
            imputed_series = series.copy() # If no imputer, return original series
            log_message = f"No imputation performed for '{column_name}'"

        return column_name, imputed_series, log_message

    

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            subset (Optional[List[str]]): A list of column names to consider when identifying duplicates.
                                          If None, all columns are considered.

        Returns:
            pd.DataFrame: The DataFrame with duplicate rows removed.
        """
        original_len = len(df)
        self._log_cleaning(f"Before duplicate removal: {original_len} rows.", reason="Recording initial row count before duplicate removal.")
        # Reset index to ensure duplicate detection is not affected by it
        df = df.reset_index(drop=True)
        df = df.drop_duplicates(subset=subset, keep='first') # Remove duplicates, keeping the first occurrence
        removed_count = original_len - len(df)
        self._log_cleaning(f"Removed {removed_count} duplicate rows. After removal: {len(df)} rows.", reason="Duplicates removed to ensure data uniqueness.")
        return df

    

    def _handle_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles data inconsistencies across object-type columns using fuzzy matching.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with inconsistencies handled.
        """
        for column in df.select_dtypes(include=['object']).columns:
            if self._is_categorical_column(df[column]):
                value_counts = df[column].value_counts()
                unique_values_sorted = value_counts.sort_values(ascending=False).index.tolist()

                mapping = {} 
                processed_values = set() 

                for canonical_val in unique_values_sorted:
                    if canonical_val in processed_values:
                        continue 

                    processed_values.add(canonical_val);

                    for other_val in unique_values_sorted:
                        if other_val == canonical_val or other_val in processed_values:
                            continue 

                        ratio = fuzz.ratio(str(canonical_val), str(other_val))
                        partial_ratio = fuzz.partial_ratio(str(canonical_val), str(other_val))
                        token_sort_ratio = fuzz.token_sort_ratio(str(canonical_val), str(other_val))

                        if max(ratio, partial_ratio, token_sort_ratio) > 80: 
                            if value_counts[canonical_val] >= value_counts[other_val]:
                                mapping[other_val] = canonical_val
                            else:
                                mapping[canonical_val] = other_val 
                            processed_values.add(other_val) 

                if mapping:
                    df[column] = df[column].replace(mapping)
                    details = f"Handled inconsistencies by mapping {len(mapping)} values."
                    self._log_cleaning(f"Handled inconsistencies in '{column}' by mapping {len(mapping)} values.", reason="Inconsistencies resolved through fuzzy matching and mapping.")

        return df

    def _is_categorical_column(self, series: pd.Series) -> bool:
        """
        Determines if a Pandas Series should be treated as a categorical column.
        Uses a heuristic based on unique value ratio and common categorical patterns.

        Args:
            series (pd.Series): The input Series.

        Returns:
            bool: True if the column is likely categorical, False otherwise.
        """
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

        # Define common patterns for categorical data (e.g., 'yes/no', 'true/false')
        categorical_patterns = [
            r'^(yes|no)',
            r'^(true|false)',
            r'^(active|inactive)',
            r'^(high|medium|low)',
            r'^(a|b|c|d|f)',
            r'^(excellent|good|fair|poor)'
        ]

        # A column is considered categorical if:
        # 1. The ratio of unique values is less than 0.5 AND the number of unique values is less than 50.
        # 2. OR, if any of the predefined categorical patterns match all values in the series.
        return (unique_ratio < 0.5 and series.nunique() < 50) or \
               any(series.astype(str).str.match(pattern).all() for pattern in categorical_patterns)

    def _count_missing_after(self, df: pd.DataFrame) -> Dict:
        """
        Counts missing values (NaN/None) for each column in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary where keys are column names and values are the count of missing values.
        """
        return df.isnull().sum().to_dict()

    def _get_outlier_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identifies columns that still contain outliers after cleaning.
        Uses the IQR method to detect outliers.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary where keys are column names with outliers, and values
                  contain the count and a sample of outliers.
        """
        outlier_info = {}
        for col in df.select_dtypes(include=[np.number]).columns: # Iterate only through numeric columns
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Identify outliers outside the IQR bounds
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                outlier_info[col] = {"count": len(outliers), "sample": outliers.head().tolist()} # Store count and a sample
        return outlier_info

    def _get_duplicate_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identifies and provides information about duplicate rows in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary containing the count of duplicate rows and a sample of them.
        """
        duplicates = df[df.duplicated(keep=False)] # Find all duplicated rows (both first and subsequent occurrences)
        return {"count": len(duplicates), "sample_rows": duplicates.head().to_dict(orient='records')} # Return count and sample

    def _get_inconsistent_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identifies columns that may still contain inconsistencies after cleaning.
        Performs a basic fuzzy matching check between unique values in object columns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary where keys are column names with potential inconsistencies,
                  and values are lists of highly similar value pairs.
        """
        inconsistency_info = {}
        for col in df.select_dtypes(include=['object']).columns: # Iterate through object (string) columns
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 1:
                for i in range(len(unique_values)):
                    for j in range(i + 1, len(unique_values)):
                        val1 = str(unique_values[i])
                        val2 = str(unique_values[j])
                        if fuzz.ratio(val1, val2) > 85: # Check for high fuzzy similarity
                            if col not in inconsistency_info:
                                inconsistency_info[col] = []
                            inconsistency_info[col].append(f"'{val1}' and '{val2}' are highly similar.")
        return inconsistency_info

    def _generate_cleaning_stats(self, df: pd.DataFrame):
        """
        Generates comprehensive statistics about the state of the DataFrame after cleaning.
        These statistics are used in the final report and by validator agents.

        Args:
            df (pd.DataFrame): The cleaned DataFrame.
        """
        self.cleaning_stats = {
            'missing_values_after': self._count_missing_after(df),
            'outliers_after': self._get_outlier_columns(df),
            'duplicates_after': self._get_duplicate_columns(df),
            'inconsistencies_after': self._get_inconsistent_columns(df),
            'final_shape': df.shape # Shape of the DataFrame after cleaning
        }

    def generate_cleaning_report(self) -> Dict:
        """
        Generates a comprehensive report summarizing the cleaning operations performed
        and the state of the data after cleaning.

        Returns:
            Dict: A dictionary representing the cleaning report.
        """
        report = {
            "summary": "Data Cleaning Report",
            "timestamp": datetime.now().isoformat(),
            "cleaning_logs": self.cleaning_logs, # Detailed log of operations
            "operation_reports": self.operation_reports # Reports for individual operations (if any)
        }
        return report