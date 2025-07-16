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
from data_purifier.utils.cached_chat_openai import CachedChatOpenAI as ChatOpenAI # For integrating with OpenAI LLMs
from sklearn.ensemble import IsolationForest # For outlier detection
from sklearn.impute import SimpleImputer # For handling missing values

from data_purifier.utils.report_generator import ReportGenerator # Utility for generating reports (though not directly used for final report here)

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
        # If an LLM is not provided, it defaults to ChatOpenAI with a temperature of 0.7 and gpt-4 model.
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory.
        # This defines the persona and capabilities of the agent within the CrewAI framework.
        self.agent = Agent(
            role="Principal Data Quality Engineer",
            goal="To meticulously analyze raw datasets, proactively identify and rectify all data quality anomalies (missing values, outliers, duplicates, inconsistencies), and apply advanced cleaning strategies to ensure the dataset is impeccably clean, reliable, and ready for sophisticated analysis, mirroring the precision of a top-tier data quality expert.",
            backstory="A veteran data quality engineer with an unparalleled eye for detail and a profound understanding of data integrity principles. Possesses extensive experience in diagnosing complex data issues, implementing robust and adaptive cleaning algorithms, and ensuring data assets meet the highest standards of accuracy and consistency for critical business intelligence and machine learning initiatives.",
            llm=self.llm, # Assign the initialized LLM to the agent
            allow_delegation=False, # This agent does not delegate tasks to others; it performs cleaning directly.
            verbose=True # Enable verbose output for the agent's actions to see its thought process.
        )
        
        # Logger for internal logging within the agent instance.
        self.logger = logging.getLogger(__name__)
        
        # Lists and dictionaries to store cleaning process details and statistics.
        self.cleaning_logs = [] # Stores a chronological log of all cleaning operations performed.
        self.cleaning_stats = {} # Stores summary statistics about the data after cleaning (e.g., remaining missing values).
        self.cleaning_config = config or {
            "max_unique_for_fuzzy_matching": 1000 # Default threshold for fuzzy matching to prevent performance issues on high-cardinality columns.
        } # Configuration specific to cleaning operations, can be overridden by external config.
        self.imputers = {} # Stores imputer models (e.g., SimpleImputer instances) for potential future use or inspection.
        self.outlier_detectors = {} # Stores outlier detector models (e.g., IsolationForest instances).
        self.operation_reports = {} # Stores detailed reports for individual cleaning operations.

        # ReportGenerator instance (currently not used for the final report, but can be extended).
        self.report_generator = ReportGenerator()
        self.cleaning_operations = [] # List to track planned cleaning operations (not actively used in the current implementation flow).

    def _log_cleaning(self, message: str, reason: str = None):
        """
        Logs a cleaning operation message with a timestamp and an optional reason.
        This is for internal tracking and console output, providing real-time feedback.

        Args:
            message (str): The descriptive message of the cleaning operation.
            reason (str, optional): The rationale or context behind performing this operation.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Get current timestamp.
        log_entry = {"timestamp": timestamp, "message": message, "reason": reason} # Create a log entry dictionary.
        self.cleaning_logs.append(log_entry) # Add the entry to the internal cleaning logs list.
        print(f"[Cleaning] {message}" + (f" (Reason: {reason})" if reason else "")) # Print to console.

    def _log_operation(self, operation: str, details: Dict):
        """
        Logs detailed information about a specific cleaning operation, including its outcome.

        Args:
            operation (str): The type of operation performed (e.g., 'handle_missing_values').
            details (Dict): A dictionary containing specific details and statistics of the operation's outcome.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Get current timestamp.
        log_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'details': details
        }
        self.cleaning_logs.append(log_entry) # Add the detailed log entry.
        print(f"[{timestamp}] {operation}: {json.dumps(details, indent=2)}") # Print to console with pretty JSON.

    def _validate_operation_result(self, df: pd.DataFrame, operation: str) -> bool:
        """
        Placeholder for calling a dedicated validator agent.
        In this architecture, the actual validation is performed by CleaningValidatorAgent
        after the CleanerAgent completes its work. This method currently serves as a stub.

        Args:
            df (pd.DataFrame): The DataFrame after an operation has been applied.
            operation (str): The name of the operation that was performed.

        Returns:
            bool: Always True, as actual validation is external and handled by a separate agent.
        """
        return True

    def _log_validation_issue(self, message: str):
        """
        Logs a warning message related to a validation issue identified during the cleaning process.

        Args:
            message (str): The message describing the validation issue.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Get current timestamp.
        self.logger.warning(f"[{timestamp}] Validation Issue: {message}") # Log the warning using the internal logger.

    def clean_dataset(self, df: pd.DataFrame, cleaning_instructions: Optional[Dict] = None, learned_optimizations: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Cleans the input DataFrame based on provided instructions and learned optimizations.
        This is the main method for the CleanerAgent, orchestrating various cleaning sub-operations.

        Args:
            df (pd.DataFrame): The input DataFrame to be cleaned.
            cleaning_instructions (Optional[Dict]): A dictionary of cleaning operations to apply.
                                                    Typically derived from MetaAnalyzerAgent's suggestions.
                                                    Expected format: {column_name: [{operation: ..., method: ...}]}
            learned_optimizations (Optional[Dict]): Optimizations learned from previous validation failures.
                                                    (Currently not fully utilized in this method, but passed for future extensibility).

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing the cleaned DataFrame and a dictionary
                                       of cleaning statistics and reports summarizing the operations.
        """
        # Update cleaning configuration with new instructions (e.g., from Orchestrator's retries).
        # This allows dynamic adjustment of cleaning parameters.
        self.cleaning_config.update(cleaning_instructions or {})
        self.cleaning_stats = {} # Reset summary statistics for the current cleaning run.
        self.operation_reports = {} # Reset individual operation reports.

        cleaned_df = df.copy() # Work on a copy of the DataFrame to avoid modifying the original input directly.
        operations_performed_details = [] # List to accumulate details of each cleaning operation performed.

        # Iterate through the cleaning instructions provided.
        # Instructions are typically structured as {target_key: [list_of_operations]},
        # where target_key can be a column name or 'global' for dataset-wide operations.
        for target, ops_list in cleaning_instructions.items():
            # Ensure the operations list is indeed a list to prevent errors.
            if not isinstance(ops_list, list):
                self.logger.warning(f"Expected a list of operations for {target}, but got {type(ops_list)}. Skipping.")
                continue

            # Apply each operation specified for the current target.
            for op_details in ops_list:
                operation_type = op_details.get('operation') # Get the type of cleaning operation.
                status = "failed" # Default status for an operation, updated upon successful completion.
                details = {} # Dictionary to store specific details and outcomes of the operation.
                column_affected = target # Initialize affected column to the target key.

                try:
                    # --- Handle Missing Values (can be applied across multiple columns) ---
                    if operation_type == 'handle_missing_values':
                        # This method processes all columns with missing values in parallel.
                        cleaned_df = self._handle_missing_values(cleaned_df, cleaning_instructions)
                        status = "completed"
                        details = "Missing values handled in parallel across relevant columns."
                        column_affected = "All relevant columns" # Indicate a global effect.

                    # --- Handle Outliers for a specific column ---
                    elif operation_type == 'handle_outliers':
                        col_name = target
                        if col_name in cleaned_df.columns: # Check if the target column exists.
                            # Apply the specified outlier handling method to the column.
                            cleaned_df[col_name], status, details = self._apply_outlier_handling(cleaned_df[col_name], op_details.get('method'))
                            column_affected = col_name
                        else:
                            status = "skipped"
                            details = f"Column {col_name} not found for outlier handling."

                    # --- Remove Duplicate Rows (global or subset-based) ---
                    elif operation_type == 'remove_duplicates':
                        subset = op_details.get('subset') # Get optional subset of columns for duplicate identification.
                        original_len = len(cleaned_df) # Store original DataFrame length.
                        
                        # If the target is 'global' or no subset is specified, apply to the entire DataFrame.
                        if target == 'global' or subset is None:
                            cleaned_df = self.remove_duplicates(cleaned_df, subset=subset)
                            removed_count = original_len - len(cleaned_df) # Calculate number of rows removed.
                            status = "completed"
                            details = f"Removed {removed_count} duplicate rows, subset: {subset}"
                        else:
                            # If a specific column is targeted for duplicate removal, it's usually a misconfiguration
                            # as duplicate removal is typically a row-wise operation.
                            status = "skipped"
                            details = f"Duplicate removal operation for specific column {target} is not supported. Use 'global' target."

                    # --- Handle Inconsistencies for a specific column (e.g., fuzzy matching) ---
                    elif operation_type == 'handle_inconsistencies':
                        col_name = target
                        # Check if fuzzy matching is enabled in the cleaning configuration.
                        if self.cleaning_config.get('enable_fuzzy_matching', True):
                            if col_name in cleaned_df.columns: # Check if the target column exists.
                                # Apply the inconsistency handling method.
                                cleaned_df[col_name], status, details = self._apply_inconsistency_handling(cleaned_df[col_name], op_details.get('method'))
                                column_affected = col_name
                            else:
                                status = "skipped"
                                details = f"Column {col_name} not found for inconsistency handling."
                        else:
                            status = "skipped"
                            details = "Fuzzy matching disabled in configuration."

                    # --- Handle Unknown Operations ---
                    else:
                        status = "skipped"
                        details = f"Unknown cleaning operation type: {operation_type}"

                except Exception as e:
                    # Catch any exceptions that occur during an operation and log them.
                    error_message = f"Error applying cleaning operation {operation_type} on {column_affected}: {e}"
                    self.logger.error(error_message, exc_info=True) # Log the error with traceback.
                    status = "failed"
                    details = str(e) # Store the error message in details.

                # Record details of the performed operation, regardless of success or failure.
                operations_performed_details.append({
                    "operation": operation_type,
                    "column": column_affected,
                    "status": status,
                    "details": details
                })

        # Generate final cleaning statistics after all operations have been attempted.
        self._generate_cleaning_stats(cleaned_df)

        # Generate a comprehensive cleaning report for internal use or logging.
        final_report = self.generate_cleaning_report()
        self.cleaning_stats['report'] = final_report # Add the full report to cleaning statistics.
        self.cleaning_stats['operation_reports'] = self.operation_reports # Add individual operation reports.
        self.cleaning_stats['operations_performed_details'] = operations_performed_details # Add detailed log of operations.

        return cleaned_df, self.cleaning_stats

    def _apply_outlier_handling(self, series: pd.Series, method: str) -> Tuple[pd.Series, str, str]:
        """
        Applies outlier handling to a single Pandas Series (column).
        Supports 'isolation_forest', 'iqr_clipping', and 'z_score_removal' methods.

        Args:
            series (pd.Series): The input Series (column) to process.
            method (str): The outlier handling method to use.

        Returns:
            Tuple[pd.Series, str, str]: The Series with outliers handled, the status of the operation,
                                       and a detailed message about the outcome.
        """
        series = series.copy() # Work on a copy to avoid modifying the original Series directly.
        status = "failed"
        details = ""
        try:
            # Outlier handling typically applies only to numeric data types.
            if pd.api.types.is_numeric_dtype(series):
                method_used = method # Keep track of the method actually applied.
                
                # --- Isolation Forest Method ---
                if method == 'isolation_forest':
                    # Initialize IsolationForest model. contamination is the expected proportion of outliers.
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    # Fit the model and predict outliers (-1 for outliers, 1 for inliers).
                    outliers = iso_forest.fit_predict(series.to_frame())
                    original_outlier_count = (outliers == -1).sum() # Count detected outliers.
                    if original_outlier_count > 0:
                        # Replace outliers with the median of the series (a common strategy).
                        series.loc[outliers == -1] = series.median()
                        status = "completed"
                        details = f"Outliers handled using {method}. Replaced {original_outlier_count} outliers with median."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method}."
                        self.logger.info(details)
                
                # --- IQR (Interquartile Range) Clipping Method ---
                elif method == 'iqr_clipping':
                    Q1 = series.quantile(0.25) # 1st quartile
                    Q3 = series.quantile(0.75) # 3rd quartile
                    IQR = Q3 - Q1 # Interquartile Range
                    lower_bound = Q1 - 1.5 * IQR # Lower bound for outlier detection
                    upper_bound = Q3 + 1.5 * IQR # Upper bound for outlier detection
                    original_outlier_count = series[(series < lower_bound) | (series > upper_bound)].count() # Count outliers.
                    if original_outlier_count > 0:
                        # Clip values to the calculated bounds, effectively capping outliers.
                        series = series.clip(lower=lower_bound, upper=upper_bound)
                        status = "completed"
                        details = f"Outliers handled using {method}. Clipped {original_outlier_count} outliers."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method}."
                        self.logger.info(details)
                
                # --- Z-Score Removal Method ---
                elif method == 'z_score_removal':
                    from scipy.stats import zscore # Import zscore function.
                    # Calculate absolute Z-scores, dropping NaN values first.
                    z_scores = np.abs(zscore(series.dropna()))
                    original_outlier_count = series[z_scores > 3].count() # Count outliers using a Z-score threshold of 3.
                    if original_outlier_count > 0:
                        # Remove rows where the Z-score exceeds the threshold.
                        series = series[z_scores <= 3] 
                        status = "completed"
                        details = f"Outliers handled using {method}. Removed {original_outlier_count} outliers."
                        self.logger.info(details)
                    else:
                        status = "completed"
                        details = f"No outliers detected in column using {method}."
                        self.logger.info(details)
                
                # --- Adaptive Choice for Numeric Columns ---
                else: 
                    # If no specific method is provided or an unsupported method is given, default to IQR clipping.
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
        Supports 'mean', 'median', and 'mode' imputation strategies.

        Args:
            series (pd.Series): The input Series (column) to impute.
            method (str): The imputation method to use ('mean', 'median', 'mode').

        Returns:
            Tuple[pd.Series, str, str]: The Series with missing values imputed, the status of the operation,
                                       and a detailed message about the outcome.
        """
        status = "failed"
        details = ""
        try:
            if series.isnull().any(): # Only proceed if there are missing values in the series.
                imputer = None # Initialize imputer to None.
                
                # --- Numeric Column Imputation ---
                if pd.api.types.is_numeric_dtype(series):
                    if method == 'mean' or method == 'mean_imputation':
                        imputer = SimpleImputer(strategy='mean')
                        method_used = 'mean'
                    elif method == 'median':
                        imputer = SimpleImputer(strategy='median')
                        method_used = 'median'
                    else: # Adaptive choice for numeric if no specific method or unsupported.
                        # If the data is highly skewed, median is a more robust imputation strategy.
                        if series.skew() > 1 or series.skew() < -1: 
                            imputer = SimpleImputer(strategy='median')
                            method_used = 'median (adaptive)'
                        else:
                            imputer = SimpleImputer(strategy='mean')
                            method_used = 'mean (adaptive)'
                
                # --- Non-Numeric (Categorical/Object) Column Imputation ---
                else: 
                    if method == 'mode' or method == 'mode_imputation':
                        imputer = SimpleImputer(strategy='most_frequent')
                        method_used = 'mode'
                    else: # Adaptive choice for non-numeric if no specific method or unsupported.
                        # For non-numeric data, the most frequent value (mode) is typically used.
                        imputer = SimpleImputer(strategy='most_frequent')
                        method_used = 'mode (adaptive)'

                if imputer:
                    # Apply imputation: reshape series to 2D array for imputer, then flatten back to 1D Series.
                    imputed_series = pd.Series(imputer.fit_transform(series.to_frame())[:, 0], index=series.index, name=series.name)
                    status = "completed"
                    details = f"Missing values handled using {method} strategy."
                    return imputed_series, status, details
                else:
                    details = f"Unknown or unsupported imputation method: {method}."
            else:
                status = "completed"
                details = "No missing values to handle." # If no NaNs, no action is needed.
        except Exception as e:
            details = str(e) # Capture any exception message.
        return series, status, details

    def _apply_inconsistency_handling(self, series: pd.Series, method: str) -> Tuple[pd.Series, str, str]:
        """
        Applies inconsistency handling to a single Pandas Series, primarily for categorical data.
        Uses fuzzy matching to identify and map similar-looking values to a canonical form.
        Supports 'standardize_case' and 'fuzzy_match' methods.

        Args:
            series (pd.Series): The input Series (column) to process.
            method (str): The inconsistency handling method to use.

        Returns:
            Tuple[pd.Series, str, str]: The Series with inconsistencies handled, the status of the operation,
                                       and a detailed message about the outcome.
        """
        status = "failed"
        details = ""
        method_used = method # To track the method actually used.
        try:
            if self._is_categorical_column(series): # Only apply to columns identified as categorical.
                # Get the maximum number of unique values for which fuzzy matching will be attempted.
                # This prevents performance issues on very high-cardinality columns.
                max_unique = self.cleaning_config.get('max_unique_for_fuzzy_matching', 1000)

                # --- Standardize Case Method ---
                if method == 'standardize_case':
                    # Convert to string, strip whitespace, and convert to lowercase.
                    series = series.astype(str).str.strip().str.lower()
                    status = "completed"
                    details = "Inconsistencies handled by standardizing case and trimming whitespace."
                
                # --- Fuzzy Match Method ---
                elif method == 'fuzzy_match':
                    if series.nunique() > max_unique: # Skip if cardinality is too high.
                        status = "skipped"
                        details = f"Skipping fuzzy matching for column '{series.name}' due to high cardinality ({series.nunique()} unique values > {max_unique})."
                        self.logger.warning(details)
                        return series, status, details

                    value_counts = series.value_counts() # Get frequency of each unique value.
                    # Sort unique values by their frequency in descending order.
                    unique_values_sorted = value_counts.sort_values(ascending=False).index.tolist()

                    mapping = {} # Dictionary to store mappings from inconsistent values to canonical values.
                    processed_values = set() # Set to keep track of values already processed as canonical or mapped.

                    for canonical_val in unique_values_sorted:
                        if canonical_val in processed_values: # Skip if already processed.
                            continue

                        processed_values.add(canonical_val); # Mark as processed.

                        for other_val in unique_values_sorted:
                            if other_val == canonical_val or other_val in processed_values: # Skip if same or already processed.
                                continue

                            # Calculate fuzzy matching ratios.
                            ratio = fuzz.ratio(str(canonical_val), str(other_val))
                            partial_ratio = fuzz.partial_ratio(str(canonical_val), str(other_val))
                            token_sort_ratio = fuzz.token_sort_ratio(str(canonical_val), str(other_val))

                            # If any ratio is above 80 (high similarity), consider them inconsistent.
                            if max(ratio, partial_ratio, token_sort_ratio) > 80: 
                                # Map the less frequent value to the more frequent one.
                                if value_counts[canonical_val] >= value_counts[other_val]:
                                    mapping[other_val] = canonical_val
                                else:
                                    mapping[canonical_val] = other_val 
                                processed_values.add(other_val) # Mark the mapped value as processed.

                    if mapping:
                        series = series.replace(mapping) # Apply the generated mapping to the Series.
                        status = "completed"
                        details = f"Handled inconsistencies by fuzzy matching and mapping {len(mapping)} values."
                    else:
                        status = "completed"
                        details = "No inconsistencies found or mapped via fuzzy matching."
                
                # --- Adaptive Choice for Inconsistency Handling ---
                else: 
                    method_used = 'adaptive_standardize_and_fuzzy'
                    # First, standardize case and trim whitespace for initial consistency.
                    original_series = series.copy()
                    series = series.astype(str).str.strip().str.lower()

                    if series.nunique() > max_unique: # Skip adaptive fuzzy matching if cardinality is too high.
                        status = "skipped"
                        details = f"Skipping adaptive fuzzy matching for column '{series.name}' due to high cardinality ({series.nunique()} unique values > {max_unique})."
                        self.logger.warning(details)
                        return original_series, status, details # Return original if skipped.

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
                        # Apply mapping to the original series, then re-apply standardized case/trim.
                        series = original_series.replace(mapping)
                        series = series.astype(str).str.strip().str.lower()
                        status = "completed"
                        details = f"Handled inconsistencies by adaptive fuzzy matching and mapping {len(mapping)} values."
                    else:
                        status = "completed"
                        details = "No inconsistencies found or mapped via adaptive fuzzy matching."
            else:
                status = "completed"
                details = "Column is not categorical for inconsistency handling." # Skip if not a categorical column.
        except Exception as e:
            details = str(e) # Capture any exception message.
        return series, status, details

    

    def _analyze_outliers_iqr(self, series: pd.Series) -> Dict:
        """
        Analyzes outliers in a numeric series using the Interquartile Range (IQR) method.
        This method calculates the IQR and identifies values falling outside 1.5 times the IQR
        from the first and third quartiles.

        Args:
            series (pd.Series): The input numeric Series.

        Returns:
            Dict: A dictionary containing outlier statistics (count, ratio of outliers to non-null values).
        """
        q1 = series.quantile(0.25) # Calculate the first quartile (25th percentile).
        q3 = series.quantile(0.75) # Calculate the third quartile (75th percentile).
        iqr = q3 - q1 # Calculate the Interquartile Range.

        lower_bound = q1 - 1.5 * iqr # Calculate the lower bound for outlier detection.
        upper_bound = q3 + 1.5 * iqr # Calculate the upper bound for outlier detection.

        # Identify values that fall outside the calculated lower and upper bounds.
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        # Return a dictionary with the count and ratio of outliers.
        stats = {
            'outliers': {
                'count': len(outliers),
                'ratio': len(outliers) / len(series.dropna()) if len(series.dropna()) > 0 else 0 # Avoid division by zero.
            }
        }
        return stats

    @staticmethod
    def _impute_column_parallel(args):
        """
        Static method to perform missing value imputation on a single column.
        Designed to be used with multiprocessing for parallel execution.
        It dynamically chooses imputation strategy (mean/median/mode) based on column type and outlier presence.

        Args:
            args (Tuple): A tuple containing (column_name, series, strategy, outlier_stats).
                          - column_name (str): Name of the column.
                          - series (pd.Series): The column data.
                          - strategy (str): Preferred imputation strategy (e.g., 'mean', 'median', 'mode').
                          - outlier_stats (Dict): Statistics about outliers in the column.

        Returns:
            Tuple: (column_name, imputed_series, log_message).
                   - column_name (str): Name of the processed column.
                   - imputed_series (pd.Series): The column data after imputation.
                   - log_message (str): A message detailing the imputation performed.
        """
        column_name, series, strategy, outlier_stats = args
        imputer = None
        log_message = ""

        if pd.api.types.is_numeric_dtype(series):
            # If a specific strategy (mean/median) is not provided, dynamically choose.
            if strategy not in ['mean', 'median']:
                # If significant outliers are present (ratio > 5%), use median for robustness.
                if outlier_stats and outlier_stats['outliers']['ratio'] > 0.05:
                    strategy = 'median' 
                    log_message = f"Using median imputation for '{column_name}' due to {outlier_stats['outliers']['ratio']:.2%} outliers"
                else:
                    strategy = 'mean' # Otherwise, use mean imputation.
                    log_message = f"Using mean imputation for '{column_name}' (no significant outliers)"
            
            # Initialize SimpleImputer based on the chosen numeric strategy.
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
        else:
            # For non-numeric columns (categorical/object), default to mode imputation.
            strategy = 'most_frequent' # Use 'most_frequent' strategy for non-numeric data.
            imputer = SimpleImputer(strategy='most_frequent')
            log_message = f"Column '{column_name}' is non-numeric, using {strategy} imputation"

        if imputer:
            # Apply imputation: fit the imputer and transform the series.
            # Reshape series to 2D array for imputer, then flatten back to 1D Series.
            imputed_series = pd.Series(imputer.fit_transform(series.to_frame())[:, 0], index=series.index, name=column_name)
            log_message += f"\nHandled missing values in '{column_name}' using {strategy} strategy"
        else:
            imputed_series = series.copy() # If no imputer was initialized, return the original series.
            log_message = f"No imputation performed for '{column_name}'"

        return column_name, imputed_series, log_message

    def _handle_missing_values(self, df: pd.DataFrame, cleaning_instructions: Dict) -> pd.DataFrame:
        """
        Handles missing values across the DataFrame.
        It identifies columns with missing values and applies appropriate imputation strategies,
        potentially leveraging parallel processing for efficiency.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cleaning_instructions (Dict): The cleaning instructions, potentially containing per-column methods.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        columns_to_impute = [] # List to store arguments for parallel imputation.
        for column in df.columns:
            if df[column].isnull().any(): # Check if the current column has any missing values.
                # Get the specific imputation strategy for this column from the cleaning_instructions.
                column_ops = cleaning_instructions.get(column, [])
                strategy = None
                for op in column_ops:
                    if op.get('operation') == 'handle_missing_values':
                        strategy = op.get('method')
                        break

                outlier_stats = None
                # Analyze outliers for numeric columns to inform adaptive imputation strategy.
                if pd.api.types.is_numeric_dtype(df[column]):
                    outlier_stats = self._analyze_outliers_iqr(df[column]) 
                
                # Add column info to the list for parallel processing.
                columns_to_impute.append((column, df[column], strategy, outlier_stats))

        if not columns_to_impute:
            return df # If no columns need imputation, return the original DataFrame.

        # Determine the number of processes to use for parallel imputation.
        # It uses the minimum of the number of columns to impute or the available CPU count.
        num_processes = min(len(columns_to_impute), multiprocessing.cpu_count()) 
        
        # Use a multiprocessing Pool to apply imputation to columns in parallel.
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the static _impute_column_parallel method to the list of columns.
            results = pool.map(CleanerAgent._impute_column_parallel, columns_to_impute) 

        imputed_df = df.copy() # Create a copy of the DataFrame to store imputed results.
        for col_name, imputed_series, log_message in results:
            imputed_df[col_name] = imputed_series # Update the column with the imputed values.
            self._log_cleaning(log_message) # Log the imputation details.
            # Placeholder for imputer details; in parallel mode, individual imputer instances are not easily stored per-column.
            self.imputers[col_name] = "Imputer details (not stored per-column in parallel mode)" 

        return imputed_df

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            subset (Optional[List[str]]): A list of column names to consider when identifying duplicates.
                                          If None, all columns are considered for duplication.

        Returns:
            pd.DataFrame: The DataFrame with duplicate rows removed.
        """
        original_len = len(df) # Get the number of rows before removing duplicates.
        self._log_cleaning(f"Before duplicate removal: {original_len} rows.", reason="Recording initial row count before duplicate removal.")
        
        # Reset the index to ensure duplicate detection is not affected by it.
        # `drop=True` prevents the old index from being added as a new column.
        df = df.reset_index(drop=True)
        
        # Remove duplicate rows, keeping the first occurrence found.
        df = df.drop_duplicates(subset=subset, keep='first') 
        
        removed_count = original_len - len(df) # Calculate the number of rows that were removed.
        self._log_cleaning(f"Removed {removed_count} duplicate rows. After removal: {len(df)} rows.", reason="Duplicates removed to ensure data uniqueness.")
        return df

    def _handle_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles data inconsistencies across object-type columns using fuzzy matching.
        This method iterates through object (string) columns and attempts to standardize
        similar-looking values to a canonical form based on fuzzy string matching.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with inconsistencies handled.
        """
        # Iterate through columns that are of 'object' dtype (typically strings).
        for column in df.select_dtypes(include=['object']).columns:
            # Check if the column is considered categorical based on a heuristic.
            if self._is_categorical_column(df[column]):
                value_counts = df[column].value_counts() # Get the frequency of each unique value.
                # Sort unique values by their frequency in descending order.
                unique_values_sorted = value_counts.sort_values(ascending=False).index.tolist()

                mapping = {} # Dictionary to store mappings from inconsistent values to canonical values.
                processed_values = set() # Set to keep track of values already processed.

                for canonical_val in unique_values_sorted:
                    if canonical_val in processed_values: # Skip if this value has already been processed.
                        continue 

                    processed_values.add(canonical_val); # Mark the current canonical value as processed.

                    for other_val in unique_values_sorted:
                        if other_val == canonical_val or other_val in processed_values: # Skip if it's the same value or already processed.
                            continue 

                        # Calculate various fuzzy matching ratios between the canonical and other values.
                        ratio = fuzz.ratio(str(canonical_val), str(other_val))
                        partial_ratio = fuzz.partial_ratio(str(canonical_val), str(other_val))
                        token_sort_ratio = fuzz.token_sort_ratio(str(canonical_val), str(other_val))

                        # If any of the fuzzy ratios are above 80 (indicating high similarity),
                        # consider them as inconsistent variations of the same value.
                        if max(ratio, partial_ratio, token_sort_ratio) > 80: 
                            # Map the less frequent value to the more frequent one to standardize.
                            if value_counts[canonical_val] >= value_counts[other_val]:
                                mapping[other_val] = canonical_val
                            else:
                                mapping[canonical_val] = other_val 
                            processed_values.add(other_val) # Mark the mapped value as processed.

                if mapping:
                    df[column] = df[column].replace(mapping) # Apply the generated mapping to the DataFrame column.
                    details = f"Handled inconsistencies by mapping {len(mapping)} values."
                    self._log_cleaning(f"Handled inconsistencies in '{column}' by mapping {len(mapping)} values.", reason="Inconsistencies resolved through fuzzy matching and mapping.")

        return df

    def _is_categorical_column(self, series: pd.Series) -> bool:
        """
        Determines if a Pandas Series should be treated as a categorical column based on heuristics.
        A column is considered categorical if its unique value ratio is low or if it matches common
        categorical patterns (e.g., 'yes/no', 'true/false').

        Args:
            series (pd.Series): The input Series (column).

        Returns:
            bool: True if the column is likely categorical, False otherwise.
        """
        # Calculate the ratio of unique values to the total number of values.
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

        # Define common patterns that often indicate categorical data.
        categorical_patterns = [
            r'^(yes|no)',
            r'^(true|false)',
            r'^(active|inactive)',
            r'^(high|medium|low)',
            r'^(a|b|c|d|f)',
            r'^(excellent|good|fair|poor)'
        ]

        # A column is considered categorical if:
        # 1. The ratio of unique values is less than 0.5 (50%) AND the number of unique values is less than 50.
        #    This heuristic captures columns with a limited number of distinct values.
        # 2. OR, if any of the predefined regular expression patterns match all values in the series.
        #    This captures common binary or ordinal categorical patterns.
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
        return df.isnull().sum().to_dict() # Returns a dictionary of column names to their null counts.

    def _get_outlier_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identifies columns that still contain outliers after cleaning using the IQR method.
        This method is used for reporting purposes to show residual outliers.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary where keys are column names with outliers, and values
                  contain the count and a sample of the outlier values.
        """
        outlier_info = {} # Initialize an empty dictionary to store outlier information.
        # Iterate only through numeric columns as outliers are typically defined for numerical data.
        for col in df.select_dtypes(include=[np.number]).columns: 
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers: values below the lower bound or above the upper bound.
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty: # If any outliers are found in the column.
                # Store the count of outliers and a sample of the outlier values.
                outlier_info[col] = {"count": len(outliers), "sample": outliers.head().tolist()} 
        return outlier_info

    def _get_duplicate_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identifies and provides information about duplicate rows in the DataFrame.
        This method is used for reporting purposes to show residual duplicates.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary containing the count of duplicate rows and a sample of them.
        """
        # Find all duplicated rows, keeping both the first and subsequent occurrences for reporting.
        duplicates = df[df.duplicated(keep=False)] 
        # Return the count of duplicate rows and a sample of these rows.
        return {"count": len(duplicates), "sample_rows": duplicates.head().to_dict(orient='records')} 

    def _get_inconsistent_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identifies columns that may still contain inconsistencies after cleaning.
        Performs a basic fuzzy matching check between unique values in object columns
        to detect highly similar but not identical entries.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Dict: A dictionary where keys are column names with potential inconsistencies,
                  and values are lists of highly similar value pairs found.
        """
        inconsistency_info = {} # Initialize an empty dictionary to store inconsistency information.
        # Iterate through object (string) columns.
        for col in df.select_dtypes(include=['object']).columns: 
            unique_values = df[col].dropna().unique() # Get unique non-null values.
            if len(unique_values) > 1: # Only check if there's more than one unique value.
                # Compare each unique value with every other unique value.
                for i in range(len(unique_values)):
                    for j in range(i + 1, len(unique_values)):
                        val1 = str(unique_values[i])
                        val2 = str(unique_values[j])
                        # Check for high fuzzy similarity (ratio > 85) between value pairs.
                        if fuzz.ratio(val1, val2) > 85: 
                            if col not in inconsistency_info:
                                inconsistency_info[col] = [] # Initialize list for column if not present.
                            inconsistency_info[col].append(f"'{val1}' and '{val2}' are highly similar.")
        return inconsistency_info

    def _generate_cleaning_stats(self, df: pd.DataFrame):
        """
        Generates comprehensive statistics about the state of the DataFrame after cleaning.
        These statistics are used in the final report and by validator agents to assess
        the effectiveness of the cleaning operations.

        Args:
            df (pd.DataFrame): The cleaned DataFrame.
        """
        self.cleaning_stats = {
            'missing_values_after': self._count_missing_after(df), # Count remaining missing values.
            'outliers_after': self._get_outlier_columns(df), # Identify remaining outliers.
            'duplicates_after': self._get_duplicate_columns(df), # Identify remaining duplicate rows.
            'inconsistencies_after': self._get_inconsistent_columns(df), # Identify remaining inconsistencies.
            'final_shape': df.shape # Record the final shape (rows, columns) of the DataFrame.
        }

    def generate_cleaning_report(self) -> Dict:
        """
        Generates a comprehensive report summarizing the cleaning operations performed
        and the state of the data after cleaning. This report is part of the overall
        pipeline execution log.

        Returns:
            Dict: A dictionary representing the cleaning report, including summary,
                  timestamp, and detailed logs.
        """
        report = {
            "summary": "Data Cleaning Report",
            "timestamp": datetime.now().isoformat(), # Current timestamp of report generation.
            "cleaning_logs": self.cleaning_logs, # Detailed chronological log of cleaning operations.
            "operation_reports": self.operation_reports # Reports for individual operations (if any).
        }
        return report
