import logging
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import contractions
import nltk
import pandas as pd
from crewai import Agent
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from langchain_openai import ChatOpenAI

from utils.report_generator import ReportGenerator

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TransformerAgent:
    def __init__(self, config: Dict = None, llm=None):
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        self.agent = Agent(
            role="Principal Data Transformation Architect",
            goal="To meticulously reshape and convert raw datasets into highly optimized and standardized formats, ensuring seamless integration and maximum utility for advanced analytical models and reporting, operating with the foresight of a lead data architect.",
            backstory="A visionary data architect with extensive experience in designing and implementing complex data transformation pipelines. Specializes in optimizing data structures, harmonizing disparate data sources, and applying advanced text processing and encoding techniques to prepare data for high-performance analytics and machine learning, ensuring data assets are perfectly aligned with strategic business objectives.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )
        self.transformation_logs = []
        self.logger = logging.getLogger(__name__)

    def _log_transformation(self, message: str, reason: str = None):
        """Log transformation operations with timestamp and optional reason"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "message": message, "reason": reason}
        self.transformation_logs.append(log_entry)
        print(f"[Transformation] {message}" + (f" (Reason: {reason})" if reason else ""))

    def transform_data(self, df: pd.DataFrame, processing_instructions: Dict, learned_optimizations: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """Transform the dataframe according to the transformation instructions."""
        transformation_report = {
            "operations_performed": []
        }
        transformed_df = df.copy()

        trans_ops = processing_instructions.get('transformation_operations', {})

        for target_key, operations_list in trans_ops.items():
            if not isinstance(operations_list, list):
                self.logger.warning(f"Expected a list of operations for {target_key}, but got {type(operations_list)}. Skipping.")
                continue

            for op_details in operations_list:
                operation_type = op_details.get('operation')
                status = "failed"
                details = {}
                column_affected = target_key # Default to target_key as column

                try:
                    if operation_type == 'scale_normalize':
                        transformed_df, status, details = self._apply_single_scaling_normalization(transformed_df, target_key, op_details)
                        column_affected = target_key
                    elif operation_type == 'pivot' or operation_type == 'unpivot':
                        transformed_df, status, details = self._apply_pivoting_unpivoting(transformed_df, op_details)
                        column_affected = "global" # Pivoting/unpivoting affects the whole DF structure
                    elif operation_type == 'merge' or operation_type == 'join':
                        transformed_df, status, details = self._apply_merging_joining(transformed_df, op_details)
                        column_affected = "global" # Merging/joining affects the whole DF structure
                    elif operation_type == 'text_processing':
                        transformed_df, status, details = self._apply_single_text_processing(transformed_df, target_key, op_details)
                        column_affected = op_details.get('new_column_name', target_key) # Could be a new column
                    elif operation_type == 'encode_categorical':
                        transformed_df, status, details = self._apply_single_categorical_encoding(transformed_df, target_key, op_details)
                        column_affected = op_details.get('new_column_name', target_key) # Could be a new column or multiple
                    elif operation_type == 'discretize_bin':
                        transformed_df, status, details = self._apply_single_discretization_binning(transformed_df, target_key, op_details)
                        column_affected = op_details.get('new_column_name', f'{target_key}_binned')
                    else:
                        self.logger.warning(f"Unknown transformation operation type: {operation_type} for {target_key}. Skipping.")
                        status = "skipped"
                        details = f"Unknown operation type: {operation_type}"

                except Exception as e:
                    error_message = f"Error applying transformation operation {operation_type} on {column_affected}: {e}"
                    self.logger.error(error_message, exc_info=True)
                    status = "failed"
                    details = str(e)

                transformation_report["operations_performed"].append({
                    "operation": operation_type,
                    "column": column_affected,
                    "status": status,
                    "details": details
                })

        return transformed_df, transformation_report

    def _apply_single_discretization_binning(self, df: pd.DataFrame, col_name: str, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """Handles a single discretization/binning operation."""
        status = "failed"
        details = {}
        method = op_config.get('method')
        bins = op_config.get('bins')
        labels = op_config.get('labels')
        new_column_name = op_config.get('new_column_name', f'{col_name}_binned')

        try:
            if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
                error_message = f"Column '{col_name}' not found or not numeric type for binning."
                self.logger.error(error_message)
                details = {"error": error_message}
                return df, status, details

            if method == 'equal_width':
                df[new_column_name] = pd.cut(df[col_name], bins=bins, labels=labels, include_lowest=True)
            elif method == 'equal_frequency':
                df[new_column_name] = pd.qcut(df[col_name], q=bins, labels=labels, duplicates='drop')
            else:
                error_message = f"Unknown binning method: {method}."
                self.logger.error(error_message)
                details = {"error": error_message}
                return df, status, details

            self._log_transformation(f"Discretized/Binned column '{col_name}' into '{new_column_name}' using {method}.", reason=f"Column {col_name} binned into {new_column_name} for categorical analysis.")
            status = "completed"
            details = {"method": method, "bins": bins, "new_column_name": new_column_name}
            return df, status, details
        except Exception as e:
            error_message = f"Binning failed for '{col_name}' with method '{method}': {e}"
            self.logger.error(error_message, exc_info=True)
            details = {"error": error_message}
            return df, status, details

    def _apply_pivoting_unpivoting(self, df: pd.DataFrame, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """Handles a single pivoting or unpivoting operation."""
        status = "failed"
        details = {}
        operation_type = op_config.get('operation')

        try:
            if operation_type == 'pivot':
                index_cols = op_config.get('index')
                columns_to_pivot = op_config.get('columns')
                values_to_aggregate = op_config.get('values')
                agg_func = op_config.get('agg_func', 'mean')

                try:
                    pivoted_df = df.pivot_table(index=index_cols, columns=columns_to_pivot, values=values_to_aggregate, aggfunc=agg_func)
                    df = pivoted_df.reset_index() # Flatten multi-index columns if created
                    self._log_transformation(f"Pivoted data with index {index_cols}, columns {columns_to_pivot}, values {values_to_aggregate}.", reason="Data pivoted to reshape for analysis.")
                    status = "completed"
                    details = {"index": index_cols, "columns": columns_to_pivot, "values": values_to_aggregate, "agg_func": agg_func}
                except Exception as e:
                    self._log_transformation(f"Pivoting failed: {e}", reason=f"Pivoting failed due to error: {str(e)}")
                    status = "failed"
                    details = {"error": str(e)}

            elif operation_type == 'unpivot':
                id_vars = op_config.get('id_vars')
                value_vars = op_config.get('value_vars')
                var_name = op_config.get('var_name', 'variable')
                value_name = op_config.get('value_name', 'value')

                try:
                    unpivoted_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
                    df = unpivoted_df
                    self._log_transformation(f"Unpivoted data with id_vars {id_vars}, value_vars {value_vars}.", reason="Data unpivoted to reshape for analysis.")
                    status = "completed"
                    details = {"id_vars": id_vars, "value_vars": value_vars}
                except Exception as e:
                    self._log_transformation(f"Unpivoting failed: {e}", reason=f"Unpivoting failed due to error: {str(e)}")
                    status = "failed"
                    details = {"error": str(e)}
        except Exception as e:
            error_message = f"Error in pivoting/unpivoting operation {operation_type}: {e}"
            self.logger.error(error_message, exc_info=True)
            status = "failed"
            details = {"error": error_message}
        return df, status, details

    def _apply_merging_joining(self, df: pd.DataFrame, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """Handles a single merging or joining operation."""
        status = "failed"
        details = {}
        operation_type = op_config.get('operation')

        try:
            if operation_type == 'merge':
                right_df_name = op_config.get('right_df_name') # Name/identifier of the other dataframe
                on_cols = op_config.get('on')
                how = op_config.get('how', 'inner')

                error_message = f"Attempting to merge with {right_df_name} on {on_cols} with {how} join. This operation requires an external DataFrame, which is not directly supported by this agent's current design. Skipping."
                self.logger.warning(error_message)
                status = "skipped"
                details = {"error": error_message}

            elif operation_type == 'join':
                error_message = "Attempting to join with another dataframe. This operation requires an external DataFrame, which is not directly supported by this agent's current design. Skipping."
                self.logger.warning(error_message)
                status = "skipped"
                details = {"error": error_message}
        except Exception as e:
            error_message = f"Error in merging/joining operation {operation_type}: {e}"
            self.logger.error(error_message, exc_info=True)
            status = "failed"
            details = {"error": error_message}
        return df, status, details

    def _apply_single_text_processing(self, df: pd.DataFrame, col_name: str, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """Handles a single text processing operation."""
        status = "failed"
        details = {}
        method = op_config.get('method')
        new_column_name = op_config.get('new_column_name', f'{col_name}_{method}')

        try:
            if col_name not in df.columns or not pd.api.types.is_string_dtype(df[col_name]):
                error_message = f"Column '{col_name}' not found or not string type for text processing."
                self.logger.error(error_message)
                details = {"error": error_message}
                return df, status, details

            if method == 'lowercase':
                df[new_column_name] = df[col_name].str.lower()
            elif method == 'remove_punctuation':
                # Pre-compile regex for performance
                if not hasattr(self, '_punctuation_re'):
                    self._punctuation_re = re.compile(r'[^\w\s]')
                df[new_column_name] = df[col_name].apply(lambda x: self._punctuation_re.sub('', str(x)))
            elif method == 'remove_stopwords':
                # Ensure stop_words and lemmatizer are initialized if needed
                if not hasattr(self, 'stop_words'):
                    self.stop_words = set(stopwords.words('english'))
                df[new_column_name] = df[col_name].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in self.stop_words]))
            elif method == 'lemmatize':
                if not hasattr(self, 'lemmatizer'):
                    self.lemmatizer = WordNetLemmatizer()
                df[new_column_name] = df[col_name].apply(lambda x: ' '.join([self.lemmatizer.lemmatize(word) for word in str(x).split()]))
            elif method == 'expand_contractions':
                df[new_column_name] = df[col_name].apply(lambda x: contractions.fix(str(x)))
            elif method == 'remove_accents':
                df[new_column_name] = df[col_name].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('utf-8'))
            else:
                error_message = f"Unknown text processing method: {method}."
                self.logger.error(error_message)
                details = {"error": error_message}
                return df, status, details

            self._log_transformation(f"Applied text processing '{method}' to column '{col_name}'. New column: {new_column_name}", reason=f"Text processing method {method} applied to {col_name}.")
            status = "completed"
            details = {"method": method, "new_column_name": new_column_name}
            return df, status, details
        except Exception as e:
            error_message = f"Text processing failed for '{col_name}' with method '{method}': {e}"
            self.logger.error(error_message, exc_info=True)
            status = "failed"
            details = {"error": error_message}
            return df, status, details

    def _apply_single_scaling_normalization(self, df: pd.DataFrame, col_name: str, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Handles a single scaling and normalization operation.

        Args:
            df (pd.DataFrame): The input DataFrame.
            col_name (str): The name of the column to scale/normalize.
            op_config (Dict): Configuration for the scaling operation.

        Returns:
            Tuple[pd.DataFrame, str, Dict]: The modified DataFrame, status, and details.
        """
        status = "failed"
        details = {}
        method = op_config.get('method')
        method_used = method # To track the method actually used

        try:
            if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
                details = {"error": f"Column '{col_name}' not found or not numeric type."}
                return df, status, details

            if method == 'min_max_scaler':
                min_val = df[col_name].min()
                max_val = df[col_name].max()
                df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
                details = {"method": method, "reason": "Min-Max scaling applied as specified."}
            elif method == 'standard_scaler':
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                df[col_name] = (df[col_name] - mean_val) / std_val
                details = {"method": method, "reason": "Standard scaling applied as specified."}
            else: # Adaptive choice if no specific method or unsupported
                # Use StandardScaler if data is approximately normal, MinMaxScaler otherwise
                if df[col_name].skew() > 1 or df[col_name].skew() < -1: # Check for high skewness
                    min_val = df[col_name].min()
                    max_val = df[col_name].max()
                    df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
                    method_used = 'min_max_scaler (adaptive)'
                    details = {"method": method_used, "reason": "Adaptive: Min-Max scaling applied due to high skewness."}
                else:
                    mean_val = df[col_name].mean()
                    std_val = df[col_name].std()
                    df[col_name] = (df[col_name] - mean_val) / std_val
                    method_used = 'standard_scaler (adaptive)'
                    details = {"method": method_used, "reason": "Adaptive: Standard scaling applied (data is approximately normal)."}

            self._log_transformation(f"Applied {method_used} to column '{col_name}'.")
            status = "completed"
            return df, status, details
        except Exception as e:
            self._log_transformation(f"Scaling/Normalization failed for '{col_name}' with method '{method_used}': {e}", reason=f"Scaling/Normalization failed due to error: {str(e)}")
            details = {"error": str(e)}
            return df, status, details

    def _apply_single_categorical_encoding(self, df: pd.DataFrame, col_name: str, op_config: Dict) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Handles a single categorical encoding operation.

        Args:
            df (pd.DataFrame): The input DataFrame.
            col_name (str): The name of the column to encode.
            op_config (Dict): Configuration for the encoding operation.

        Returns:
            Tuple[pd.DataFrame, str, Dict]: The modified DataFrame, status, and details.
        """
        status = "failed"
        details = {}
        method = op_config.get('method')
        method_used = method # To track the method actually used

        if col_name not in df.columns or (not isinstance(df[col_name].dtype, pd.CategoricalDtype) and not pd.api.types.is_object_dtype(df[col_name])):
            details = {"error": f"Column '{col_name}' not found or not categorical/object type."}
            return df, status, details

        try:
            if method == 'one_hot':
                encoded_df = pd.get_dummies(df[col_name], prefix=col_name)
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[col_name])
                self._log_transformation(f"One-hot encoded column '{col_name}'.", reason="One-hot encoding applied to convert categorical data to numerical.")
                status = "completed"
                details = {"method": method, "new_columns": list(encoded_df.columns)}
            elif method == 'label_encode':
                encoder = LabelEncoder()
                df[f'{col_name}_encoded'] = encoder.fit_transform(df[col_name])
                self._log_transformation(f"Label encoded column '{col_name}'. New column: {col_name}_encoded", reason="Label encoding applied to convert categorical data to numerical.")
                status = "completed"
                details = {"method": method, "new_column": f'{col_name}_encoded'}
            elif method == 'frequency_encode':
                freq_map = df[col_name].value_counts(normalize=True)
                df[f'{col_name}_freq_encoded'] = df[col_name].map(freq_map)
                self._log_transformation(f"Frequency encoded column '{col_name}'. New column: {col_name}_freq_encoded", reason="Frequency encoding applied to convert categorical data to numerical.")
                status = "completed"
                details = {"method": method, "new_column": f'{col_name}_freq_encoded'}
            else: # Adaptive choice if no specific method or unsupported
                # Adaptive encoding based on cardinality
                unique_values_count = df[col_name].nunique()
                if unique_values_count <= 5: # Low cardinality: One-Hot Encoding
                    encoded_df = pd.get_dummies(df[col_name], prefix=col_name)
                    df = pd.concat([df, encoded_df], axis=1)
                    df = df.drop(columns=[col_name])
                    method_used = 'one_hot (adaptive)'
                    status = "completed"
                    details = {"method": method_used, "new_columns": list(encoded_df.columns), "reason": "Adaptive: One-Hot encoding applied due to low cardinality."}
                elif unique_values_count <= 50: # Medium cardinality: Label Encoding
                    encoder = LabelEncoder()
                    df[f'{col_name}_encoded'] = encoder.fit_transform(df[col_name])
                    method_used = 'label_encode (adaptive)'
                    status = "completed"
                    details = {"method": method_used, "new_column": f'{col_name}_encoded', "reason": "Adaptive: Label encoding applied due to medium cardinality."}
                else: # High cardinality: Frequency Encoding
                    freq_map = df[col_name].value_counts(normalize=True)
                    df[f'{col_name}_freq_encoded'] = df[col_name].map(freq_map)
                    method_used = 'frequency_encode (adaptive)'
                    status = "completed"
                    details = {"method": method_used, "new_column": f'{col_name}_freq_encoded', "reason": "Adaptive: Frequency encoding applied due to high cardinality."}

            self._log_transformation(f"Applied {method_used} to column '{col_name}'.")
            return df, status, details
        except Exception as e:
            self._log_transformation(f"Categorical encoding failed for '{col_name}' with method '{method_used}': {e}", reason=f"Categorical encoding failed due to error: {str(e)}")
            details = {"error": str(e)}
            return df, status, details
