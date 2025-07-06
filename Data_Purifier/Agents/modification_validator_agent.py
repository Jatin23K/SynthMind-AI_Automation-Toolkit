# agents/modification_validator_agent.py
# This agent is responsible for validating the output of the DataModifierAgent.
# It checks if feature engineering, aggregation, scaling, and binning operations
# have been applied correctly and as intended.

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from langchain_openai import ChatOpenAI # For integrating with OpenAI LLMs
from crewai import Agent # Used for defining the agent's role, goal, and backstory


class ModificationValidatorAgent:
    """
    Agent for validating the modification operations performed on a DataFrame.
    It ensures that new features are created correctly, aggregations are applied,
    and scaling/binning operations yield expected results.
    """

    def __init__(self, config: Dict = None, llm=None):
        """
        Initializes the ModificationValidatorAgent.

        Args:
            config (Dict, optional): Configuration dictionary for the agent.
            llm (optional): Language Model instance to be used by the agent.
                            Defaults to ChatOpenAI with gpt-4 if not provided.
        """
        # Initialize the LLM for the agent's reasoning capabilities
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory
        self.agent = Agent(
            role="Senior Feature Validation Specialist",
            goal="To meticulously validate the integrity and analytical utility of all modified and newly engineered features, ensuring they precisely meet design specifications and enhance downstream model performance, operating with the rigor of a lead data quality engineer.",
            backstory="A highly analytical and detail-oriented data quality engineer with deep expertise in feature validation and data modeling. Specializes in scrutinizing the outcomes of feature engineering, aggregation, scaling, and binning operations, providing critical insights and actionable feedback to guarantee that modified data assets are robust, accurate, and optimized for advanced analytics and machine learning.",
            llm=self.llm,
            allow_delegation=False, # This agent does not delegate tasks to others
            verbose=True # Enable verbose output for the agent's actions
        )
        
        # Logger for internal logging within the agent
        self.logger = logging.getLogger(__name__)

    def validate_modification(self, df: pd.DataFrame, modification_instructions: Dict) -> Tuple[Dict, bool, Dict]:
        """
        Validates the modification operations performed on the DataFrame.
        Checks for successful feature engineering, aggregation, scaling, and binning.

        Args:
            df (pd.DataFrame): The DataFrame after modification operations have been applied.
            modification_instructions (Dict): The instructions that were given to the DataModifierAgent.

        Returns:
            Tuple[Dict, bool, Dict]: 
                - validation_report (Dict): A dictionary detailing the validation outcome and issues found.
                - validation_success (bool): True if validation passed, False otherwise.
                - recommendations (Dict): A dictionary of suggested modification operations if issues are found.
        """
        validation_report = {
            "status": "success",
            "issues_found": {}, # Dictionary to store details of any issues identified
            "summary": "Modification validation passed." # Overall summary message
        }
        validation_success = True # Flag to indicate overall validation success
        recommendations = {"modification_operations": {}} # Dictionary to store recommendations for re-modification

        self.logger.info("Starting modification validation...")

        # --- 1. Validate Feature Engineering ---
        # Checks if new features were created as expected and if they contain meaningful data.
        fe_instructions = modification_instructions.get('feature_engineering', {})
        for col_name, ops in fe_instructions.items():
            for op in ops:
                if op.get('operation') == 'create_new_feature':
                    new_feature_name = op.get('new_feature_name', f'engineered_{col_name}')
                    
                    # Check 1: Was the new feature column actually created?
                    if new_feature_name not in df.columns:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["feature_engineering_missing"] = f"Expected feature '{new_feature_name}' not found."
                        validation_report["summary"] = "Modification validation failed: Feature engineering issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Expected feature '{new_feature_name}' not found after modification.")
                        if new_feature_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_feature_name] = []
                        recommendations["modification_operations"][new_feature_name].append({
                            "operation": "create_new_feature",
                            "reason": f"Feature '{new_feature_name}' was not created. Re-check formula and source columns."
                        })
                    # Check 2: Is the new feature column all nulls (indicating a potential formula error)?
                    elif df[new_feature_name].isnull().all():
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["feature_engineering_null"] = f"Feature '{new_feature_name}' is all nulls."
                        validation_report["summary"] = "Modification validation failed: Feature engineering issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Feature '{new_feature_name}' is all nulls after modification.")
                        if new_feature_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_feature_name] = []
                        recommendations["modification_operations"][new_feature_name].append({
                            "operation": "create_new_feature",
                            "reason": f"Feature '{new_feature_name}' is all nulls. Re-check formula and source columns."
                        })
                    # Check 3: Is the new feature column constant (indicating a potential formula error or trivial feature)?
                    elif df[new_feature_name].nunique() <= 1 and len(df[new_feature_name]) > 1: # Check for constant values in non-empty columns
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["feature_engineering_constant"] = f"Feature '{new_feature_name}' is constant after modification."
                        validation_report["summary"] = "Modification validation failed: Feature engineering issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Feature '{new_feature_name}' is constant after modification.")
                        if new_feature_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_feature_name] = []
                        recommendations["modification_operations"][new_feature_name].append({
                            "operation": "create_new_feature",
                            "reason": f"Feature '{new_feature_name}' is constant. Re-check formula and source columns."
                        })

        # --- 2. Validate Data Aggregation ---
        # A basic check to see if aggregation likely occurred (row count reduction).
        agg_instructions = modification_instructions.get('data_aggregation', {})
        if agg_instructions:
            # This is a very loose check and assumes aggregation always reduces rows.
            # A more robust check would compare against a stored original row count or expected group counts.
            # For now, we assume that if aggregation instructions were present, the row count should decrease.
            # The 'initial_df_rows' would need to be passed from the Orchestrator or a previous stage.
            initial_rows = validation_report.get("initial_df_rows") # Placeholder: This would need to be set by Orchestrator
            if initial_rows and len(df) >= initial_rows: # If row count did not decrease as expected
                validation_report["status"] = "failed"
                validation_report["issues_found"]["data_aggregation_rows"] = f"DataFrame row count ({len(df)}) did not decrease after aggregation."
                validation_report["summary"] = "Modification validation failed: Aggregation issue."
                validation_success = False
                self.logger.warning("Validation Issue: DataFrame row count did not decrease after aggregation.")
                if "global" not in recommendations["modification_operations"]:
                    recommendations["modification_operations"]["global"] = []
                recommendations["modification_operations"]["global"].append({
                    "operation": "aggregate",
                    "reason": "Aggregation did not reduce row count as expected. Re-check aggregation logic."
                })

        # --- 3. Validate Scaling/Normalization ---
        # Checks if numeric columns have been scaled to their expected ranges/distributions.
        scale_instructions = modification_instructions.get('scaling_normalization', {})
        for col_name, ops in scale_instructions.items():
            for op in ops:
                if op.get('operation') == 'scale_normalize':
                    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                        method = op.get('method')
                        if method == 'min_max_scaler':
                            # Min-Max scaling should result in values between 0 and 1
                            if not (df[col_name].min() >= -0.01 and df[col_name].max() <= 1.01): # Allow small floating point deviations
                                validation_report["status"] = "failed"
                                validation_report["issues_found"]["scaling_range"] = f"Column '{col_name}' not scaled to 0-1 range (min: {df[col_name].min():.2f}, max: {df[col_name].max():.2f})."
                                validation_report["summary"] = "Modification validation failed: Scaling issue."
                                validation_success = False
                                self.logger.warning(f"Validation Issue: Column '{col_name}' not scaled to 0-1 range.")
                                if col_name not in recommendations["modification_operations"]:
                                    recommendations["modification_operations"][col_name] = []
                                recommendations["modification_operations"][col_name].append({
                                    "operation": "scale_normalize",
                                    "method": method,
                                    "reason": f"Column '{col_name}' not scaled to 0-1 range. Re-run scaling."
                                })
                        elif method == 'standard_scaler':
                            # Standard scaling should result in mean close to 0 and std close to 1
                            if not (abs(df[col_name].mean()) < 0.01 and abs(df[col_name].std() - 1) < 0.01): # Allow small deviations
                                validation_report["status"] = "failed"
                                validation_report["issues_found"]["scaling_distribution"] = f"Column '{col_name}' not standardized (mean: {df[col_name].mean():.2f}, std: {df[col_name].std():.2f})."
                                validation_report["summary"] = "Modification validation failed: Scaling issue."
                                validation_success = False
                                self.logger.warning(f"Validation Issue: Column '{col_name}' not standardized.")
                                if col_name not in recommendations["modification_operations"]:
                                    recommendations["modification_operations"][col_name] = []
                                recommendations["modification_operations"][col_name].append({
                                    "operation": "scale_normalize",
                                    "method": method,
                                    "reason": f"Column '{col_name}' not standardized. Re-run scaling."
                                })

        # --- 4. Validate Discretization/Binning ---
        # Checks if binning created the new column with the correct type and number of bins.
        bin_instructions = modification_instructions.get('discretization_binning', {})
        for col_name, ops in bin_instructions.items():
            for op in ops:
                if op.get('operation') == 'discretize_bin':
                    new_column_name = op.get('new_column_name', f'{col_name}_binned')
                    
                    # Check 1: Was the binned column created?
                    if new_column_name not in df.columns:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["binning_missing"] = f"Expected binned feature '{new_column_name}' not found."
                        validation_report["summary"] = "Modification validation failed: Binning issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Expected binned feature '{new_column_name}' not found after modification.")
                        if new_column_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_column_name] = []
                        recommendations["modification_operations"][new_column_name].append({
                            "operation": "discretize_bin",
                            "reason": f"Binned feature '{new_column_name}' not found. Re-check binning logic."
                        })
                    # Check 2: Is the binned column of categorical or object type?
                    elif not (pd.api.types.is_categorical_dtype(df[new_column_name]) or pd.api.types.is_object_dtype(df[new_column_name])):
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["binning_type"] = f"Binned feature '{new_column_name}' is not categorical/object type."
                        validation_report["summary"] = "Modification validation failed: Binning issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Binned feature '{new_column_name}' is not categorical/object type.")
                        if new_column_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_column_name] = []
                        recommendations["modification_operations"][new_column_name].append({
                            "operation": "discretize_bin",
                            "reason": f"Binned feature '{new_column_name}' is not categorical/object type. Re-check binning logic."
                        })

                    # Optional: Check number of unique bins if expected_bins is an integer
                    expected_bins = op.get('bins')
                    if isinstance(expected_bins, int) and df[new_column_name].nunique() != expected_bins:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["binning_count"] = f"Binned feature '{new_column_name}' has {df[new_column_name].nunique()} unique values, expected {expected_bins}."
                        validation_report["summary"] = "Modification validation failed: Binning issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Binned feature '{new_column_name}' has unexpected number of unique values.")
                        if new_column_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_column_name] = []
                        recommendations["modification_operations"][new_column_name].append({
                            "operation": "discretize_bin",
                            "reason": f"Binned feature '{new_column_name}' has unexpected number of unique values. Re-check binning logic."
                        })

        self.logger.info(f"Modification validation finished with status: {validation_report['status']}")
        return validation_report, validation_success, recommendations