# agents/modification_validator_agent.py
# This agent is responsible for validating the output of the DataModifierAgent.
# It checks if feature engineering, aggregation, scaling, and binning operations
# have been applied correctly and as intended.

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from data_purifier.utils.cached_chat_openai import CachedChatOpenAI as ChatOpenAI # For integrating with OpenAI LLMs.
from crewai import Agent # Used for defining the agent's role, goal, and backstory within the CrewAI framework.


class ModificationValidatorAgent:
    """
    Agent for validating the modification operations performed on a DataFrame.
    It ensures that new features are created correctly, aggregations are applied,
    and scaling/binning operations yield expected results, providing feedback
    for iterative refinement.
    """

    def __init__(self, config: Dict = None, llm=None):
        """
        Initializes the ModificationValidatorAgent.

        Args:
            config (Dict, optional): Configuration dictionary for the agent. Not extensively used in this class,
                                     but passed for consistency and potential future use.
            llm (optional): Language Model instance to be used by the agent for reasoning or generating recommendations.
                            Defaults to ChatOpenAI with gpt-4 if not provided.
        """
        # Initialize the LLM for the agent's reasoning capabilities.
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory.
        self.agent = Agent(
            role="Senior Feature Validation Specialist",
            goal="To meticulously validate the integrity and analytical utility of all modified and newly engineered features, ensuring they precisely meet design specifications and enhance downstream model performance, operating with the rigor of a lead data quality engineer.",
            backstory="A highly analytical and detail-oriented data quality engineer with deep expertise in feature validation and data modeling. Specializes in scrutinizing the outcomes of feature engineering, aggregation, scaling, and binning operations, providing critical insights and actionable feedback to guarantee that modified data assets are robust, accurate, and optimized for advanced analytics and machine learning.",
            llm=self.llm,
            allow_delegation=False, # This agent does not delegate tasks; it performs validation directly.
            verbose=True # Enable verbose output for the agent's actions.
        )
        
        # Logger for internal logging within the agent instance.
        self.logger = logging.getLogger(__name__)

    def validate_modification(self, df: pd.DataFrame, modification_instructions: Dict) -> Tuple[Dict, bool, Dict]:
        """
        Validates the modification operations performed on the DataFrame.
        Checks for successful feature engineering, aggregation, scaling, and binning
        by examining the resulting DataFrame against the original instructions.

        Args:
            df (pd.DataFrame): The DataFrame after modification operations have been applied by the DataModifierAgent.
            modification_instructions (Dict): The instructions that were originally given to the DataModifierAgent.
                                          This dictionary guides the validation process.

        Returns:
            Tuple[Dict, bool, Dict]: 
                - validation_report (Dict): A dictionary detailing the validation outcome and any issues found.
                - validation_success (bool): True if all validation checks passed, False otherwise.
                - recommendations (Dict): A dictionary of suggested modification operations if issues are found,
                                          structured to be consumable by the Orchestrator for re-modification.
        """
        # Initialize the validation report with a success status, assuming no issues initially.
        validation_report = {
            "status": "success",
            "issues_found": {}, # Dictionary to store details of any specific issues identified.
            "summary": "Modification validation passed." # Overall summary message, updated if issues are found.
        }
        validation_success = True # Flag to track the overall success of the validation process.
        recommendations = {"modification_operations": {}} # Dictionary to store recommendations for re-modification.

        self.logger.info("Starting modification validation...")

        # --- 1. Validate Feature Engineering ---
        # Checks if new features were created as expected and if they contain meaningful data.
        fe_instructions = modification_instructions.get('feature_engineering', {}) # Get feature engineering instructions.
        for col_name, ops in fe_instructions.items():
            for op in ops:
                if op.get('operation') == 'create_new_feature':
                    new_feature_name = op.get('new_feature_name', f'engineered_{col_name}') # Get the expected new feature name.
                    
                    # Check 1: Was the new feature column actually created in the DataFrame?
                    if new_feature_name not in df.columns:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["feature_engineering_missing"] = f"Expected feature '{new_feature_name}' not found."
                        validation_report["summary"] = "Modification validation failed: Feature engineering issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Expected feature '{new_feature_name}' not found after modification.")
                        # Suggest re-running the feature creation.
                        if new_feature_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_feature_name] = []
                        recommendations["modification_operations"][new_feature_name].append({
                            "operation": "create_new_feature",
                            "reason": f"Feature '{new_feature_name}' was not created. Re-check formula and source columns."
                        })
                    # Check 2: Is the new feature column entirely composed of nulls?
                    # This often indicates an error in the formula or source data.
                    elif df[new_feature_name].isnull().all():
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["feature_engineering_null"] = f"Feature '{new_feature_name}' is all nulls."
                        validation_report["summary"] = "Modification validation failed: Feature engineering issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Feature '{new_feature_name}' is all nulls after modification.")
                        # Suggest re-checking the formula.
                        if new_feature_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_feature_name] = []
                        recommendations["modification_operations"][new_feature_name].append({
                            "operation": "create_new_feature",
                            "reason": f"Feature '{new_feature_name}' is all nulls. Re-check formula and source columns."
                        })
                    # Check 3: Is the new feature column constant (i.e., has only one unique value)?
                    # A constant feature might indicate a trivial calculation or an error in the formula.
                    elif df[new_feature_name].nunique() <= 1 and len(df[new_feature_name]) > 1: # Ensure it's not an empty column.
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["feature_engineering_constant"] = f"Feature '{new_feature_name}' is constant after modification."
                        validation_report["summary"] = "Modification validation failed: Feature engineering issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Feature '{new_feature_name}' is constant after modification.")
                        # Suggest re-checking the formula.
                        if new_feature_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_feature_name] = []
                        recommendations["modification_operations"][new_feature_name].append({
                            "operation": "create_new_feature",
                            "reason": f"Feature '{new_feature_name}' is constant. Re-check formula and source columns."
                        })

        # --- 2. Validate Data Aggregation ---
        # A basic check to see if aggregation likely occurred (row count reduction).
        agg_instructions = modification_instructions.get('data_aggregation', {}) # Get aggregation instructions.
        if agg_instructions:
            # This is a very loose check and assumes aggregation always reduces rows.
            # A more robust check would compare against a stored original row count or expected group counts.
            # The 'initial_df_rows' would ideally be passed from the Orchestrator or a previous stage for a precise check.
            initial_rows = validation_report.get("initial_df_rows") # Placeholder for initial row count.
            if initial_rows and len(df) >= initial_rows: # If row count did not decrease as expected after aggregation.
                validation_report["status"] = "failed"
                validation_report["issues_found"]["data_aggregation_rows"] = f"DataFrame row count ({len(df)}) did not decrease after aggregation."
                validation_report["summary"] = "Modification validation failed: Aggregation issue."
                validation_success = False
                self.logger.warning("Validation Issue: DataFrame row count did not decrease after aggregation.")
                # Suggest re-running the aggregation.
                if "global" not in recommendations["modification_operations"]:
                    recommendations["modification_operations"]["global"] = []
                recommendations["modification_operations"]["global"].append({
                    "operation": "aggregate",
                    "reason": "Aggregation did not reduce row count as expected. Re-check aggregation logic."
                })

        # --- 3. Validate Scaling/Normalization ---
        # Checks if numeric columns have been scaled to their expected ranges/distributions.
        scale_instructions = modification_instructions.get('scaling_normalization', {}) # Get scaling instructions.
        for col_name, ops in scale_instructions.items():
            for op in ops:
                if op.get('operation') == 'scale_normalize':
                    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]): # Ensure column exists and is numeric.
                        method = op.get('method')
                        if method == 'min_max_scaler':
                            # Min-Max scaling should transform values to be between 0 and 1.
                            # Allow for small floating point deviations.
                            if not (df[col_name].min() >= -0.01 and df[col_name].max() <= 1.01): 
                                validation_report["status"] = "failed"
                                validation_report["issues_found"]["scaling_range"] = f"Column '{col_name}' not scaled to 0-1 range (min: {df[col_name].min():.2f}, max: {df[col_name].max():.2f})."
                                validation_report["summary"] = "Modification validation failed: Scaling issue."
                                validation_success = False
                                self.logger.warning(f"Validation Issue: Column '{col_name}' not scaled to 0-1 range.")
                                # Suggest re-running the scaling.
                                if col_name not in recommendations["modification_operations"]:
                                    recommendations["modification_operations"][col_name] = []
                                recommendations["modification_operations"][col_name].append({
                                    "operation": "scale_normalize",
                                    "method": method,
                                    "reason": f"Column '{col_name}' not scaled to 0-1 range. Re-run scaling."
                                })
                        elif method == 'standard_scaler':
                            # Standard scaling should result in a mean close to 0 and standard deviation close to 1.
                            # Allow for small floating point deviations.
                            if not (abs(df[col_name].mean()) < 0.01 and abs(df[col_name].std() - 1) < 0.01): 
                                validation_report["status"] = "failed"
                                validation_report["issues_found"]["scaling_distribution"] = f"Column '{col_name}' not standardized (mean: {df[col_name].mean():.2f}, std: {df[col_name].std():.2f})."
                                validation_report["summary"] = "Modification validation failed: Scaling issue."
                                validation_success = False
                                self.logger.warning(f"Validation Issue: Column '{col_name}' not standardized.")
                                # Suggest re-running the scaling.
                                if col_name not in recommendations["modification_operations"]:
                                    recommendations["modification_operations"][col_name] = []
                                recommendations["modification_operations"][col_name].append({
                                    "operation": "scale_normalize",
                                    "method": method,
                                    "reason": f"Column '{col_name}' not standardized. Re-run scaling."
                                })

        # --- 4. Validate Discretization/Binning ---
        # Checks if binning created the new column with the correct type and number of bins.
        bin_instructions = modification_instructions.get('discretization_binning', {}) # Get binning instructions.
        for col_name, ops in bin_instructions.items():
            for op in ops:
                if op.get('operation') == 'discretize_bin':
                    new_column_name = op.get('new_column_name', f'{col_name}_binned') # Get the expected new binned column name.
                    
                    # Check 1: Was the binned column created in the DataFrame?
                    if new_column_name not in df.columns:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["binning_missing"] = f"Expected binned feature '{new_column_name}' not found."
                        validation_report["summary"] = "Modification validation failed: Binning issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Expected binned feature '{new_column_name}' not found after modification.")
                        # Suggest re-running the binning.
                        if new_column_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_column_name] = []
                        recommendations["modification_operations"][new_column_name].append({
                            "operation": "discretize_bin",
                            "reason": f"Binned feature '{new_column_name}' not found. Re-check binning logic."
                        })
                    # Check 2: Is the binned column of categorical or object type?
                    # Binned columns should typically be non-numeric (categorical or object).
                    elif not (pd.api.types.is_categorical_dtype(df[new_column_name]) or pd.api.types.is_object_dtype(df[new_column_name])):
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["binning_type"] = f"Binned feature '{new_column_name}' is not categorical/object type."
                        validation_report["summary"] = "Modification validation failed: Binning issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Binned feature '{new_column_name}' is not categorical/object type.")
                        # Suggest re-checking the binning logic.
                        if new_column_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_column_name] = []
                        recommendations["modification_operations"][new_column_name].append({
                            "operation": "discretize_bin",
                            "reason": f"Binned feature '{new_column_name}' is not categorical/object type. Re-check binning logic."
                        })

                    # Optional: Check if the number of unique bins matches the expected number (if provided as an integer).
                    expected_bins = op.get('bins')
                    if isinstance(expected_bins, int) and df[new_column_name].nunique() != expected_bins:
                        validation_report["status"] = "failed"
                        validation_report["issues_found"]["binning_count"] = f"Binned feature '{new_column_name}' has {df[new_column_name].nunique()} unique values, expected {expected_bins}."
                        validation_report["summary"] = "Modification validation failed: Binning issue."
                        validation_success = False
                        self.logger.warning(f"Validation Issue: Binned feature '{new_column_name}' has unexpected number of unique values.")
                        # Suggest re-checking the binning logic.
                        if new_column_name not in recommendations["modification_operations"]:
                            recommendations["modification_operations"][new_column_name] = []
                        recommendations["modification_operations"][new_column_name].append({
                            "operation": "discretize_bin",
                            "reason": f"Binned feature '{new_column_name}' has unexpected number of unique values. Re-check binning logic."
                        })

        self.logger.info(f"Modification validation finished with status: {validation_report['status']}")
        return validation_report, validation_success, recommendations
