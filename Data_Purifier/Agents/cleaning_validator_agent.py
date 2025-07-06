# agents/cleaning_validator_agent.py
# This agent is responsible for validating the output of the CleanerAgent.
# It checks for residual data quality issues and provides recommendations for further cleaning.

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz # For checking inconsistencies
from langchain_openai import ChatOpenAI # For integrating with OpenAI LLMs
from crewai import Agent # Used for defining the agent's role, goal, and backstory


class CleaningValidatorAgent:
    """
    Agent for validating the cleaning operations performed on a DataFrame.
    It identifies remaining data quality issues and suggests further actions.
    """

    def __init__(self, config: Dict = None, llm=None):
        """
        Initializes the CleaningValidatorAgent.

        Args:
            config (Dict, optional): Configuration dictionary for the agent.
            llm (optional): Language Model instance to be used by the agent.
                            Defaults to ChatOpenAI with gpt-4 if not provided.
        """
        # Initialize the LLM for the agent's reasoning capabilities
        self.llm = llm if llm else ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        # Define the CrewAI Agent with its specific role, goal, and backstory
        self.agent = Agent(
            role="Chief Data Integrity Auditor",
            goal="To conduct exhaustive audits of cleaned datasets, pinpointing any residual data quality issues with forensic precision, and generating highly specific, actionable recommendations to achieve absolute data integrity and analytical readiness, operating as a lead data quality assurance specialist.",
            backstory="A seasoned and highly meticulous data auditor with a forensic approach to data quality. Possesses an unparalleled ability to uncover hidden anomalies, validate the efficacy of cleaning processes, and provide expert-level guidance for iterative data refinement, ensuring that every dataset meets the most stringent quality benchmarks for critical business operations.",
            llm=self.llm,
            allow_delegation=False, # This agent does not delegate tasks to others
            verbose=True # Enable verbose output for the agent's actions
        )
        
        # Logger for internal logging within the agent
        self.logger = logging.getLogger(__name__)

    def validate_cleaning(self, df: pd.DataFrame, cleaning_instructions: Dict) -> Tuple[Dict, bool, Dict]:
        """
        Validates the cleaning operations performed on the DataFrame.
        Checks for remaining missing values, duplicates, outliers, and inconsistencies.

        Args:
            df (pd.DataFrame): The DataFrame after cleaning operations have been applied.
            cleaning_instructions (Dict): The instructions that were given to the CleanerAgent.
                                          (Currently not directly used for validation logic, but for context).

        Returns:
            Tuple[Dict, bool, Dict]: 
                - validation_report (Dict): A dictionary detailing the validation outcome and issues found.
                - validation_success (bool): True if validation passed, False otherwise.
                - recommendations (Dict): A dictionary of suggested cleaning operations if issues are found.
        """
        validation_report = {
            "status": "success",
            "issues_found": {}, # Dictionary to store details of any issues identified
            "summary": "Cleaning validation passed." # Overall summary message
        }
        validation_success = True # Flag to indicate overall validation success
        recommendations = {"cleaning_operations": {}} # Dictionary to store recommendations for re-cleaning

        self.logger.info("Starting cleaning validation...")
        self.logger.info(f"DataFrame shape before duplicate check: {df.shape}")
        self.logger.info(f"Number of duplicates found by validator: {df.duplicated().sum()}")

        # --- 1. Validate Missing Values ---
        # Count missing values in each column after cleaning
        missing_values_after = df.isnull().sum().to_dict()
        # Filter to get only columns that still have missing values
        remaining_missing = {col: count for col, count in missing_values_after.items() if count > 0}
        
        if remaining_missing:
            validation_report["status"] = "failed"
            validation_report["issues_found"]["missing_values"] = remaining_missing
            validation_report["summary"] = "Cleaning validation failed: Missing values still present."
            validation_success = False
            self.logger.warning(f"Validation Issue: Missing values found after cleaning: {remaining_missing}")
            
            # Generate recommendations for handling remaining missing values
            for col, count in remaining_missing.items():
                if col not in recommendations["cleaning_operations"]:
                    recommendations["cleaning_operations"][col] = []
                # Suggest imputation method based on column data type
                if pd.api.types.is_numeric_dtype(df[col]):
                    recommendations["cleaning_operations"][col].append({
                        "operation": "handle_missing_values",
                        "method": "median", # Default to median for numeric columns (robust to outliers)
                        "reason": f"Still {count} missing values. Consider median imputation."
                    })
                else:
                    recommendations["cleaning_operations"][col].append({
                        "operation": "handle_missing_values",
                        "method": "mode", # Default to mode for non-numeric columns
                        "reason": f"Still {count} missing values. Consider mode imputation."
                    })

        # --- 2. Validate Duplicates ---
        # Count duplicate rows in the DataFrame
        duplicates_after = df.duplicated().sum()
        if duplicates_after > 0:
            validation_report["status"] = "failed"
            validation_report["issues_found"]["duplicates"] = int(duplicates_after)
            validation_report["summary"] = "Cleaning validation failed: Duplicates still present."
            validation_success = False
            self.logger.warning(f"Validation Issue: {duplicates_after} duplicates found after cleaning.")
            
            # Generate recommendation for removing duplicates
            if "global" not in recommendations["cleaning_operations"]:
                recommendations["cleaning_operations"]["global"] = [] # Use "global" key for dataset-wide operations
            recommendations["cleaning_operations"]["global"].append({
                "operation": "remove_duplicates",
                "reason": f"Still {duplicates_after} duplicate rows. Re-run duplicate removal."
            })

        # --- 3. Validate Outliers ---
        # Check for outliers using the Interquartile Range (IQR) method for numeric columns.
        outlier_issues = {}
        for col in df.select_dtypes(include=[np.number]).columns: # Iterate only through numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Identify values outside the IQR bounds
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                outlier_issues[col] = {"count": len(outliers), "sample": outliers.head().tolist()}

        if outlier_issues:
            validation_report["status"] = "failed"
            validation_report["issues_found"]["outliers"] = outlier_issues
            validation_report["summary"] = "Cleaning validation failed: Outliers still present."
            validation_success = False
            self.logger.warning(f"Validation Issue: Outliers found after cleaning: {outlier_issues}")
            
            # Generate recommendations for handling outliers
            for col, details in outlier_issues.items():
                if col not in recommendations["cleaning_operations"]:
                    recommendations["cleaning_operations"][col] = []
                recommendations["cleaning_operations"][col].append({
                    "operation": "handle_outliers",
                    "method": "isolation_forest", # Default to Isolation Forest for outlier treatment
                    "reason": f"Still {details['count']} outliers. Consider Isolation Forest."
                })

        # --- 4. Validate Inconsistencies ---
        # Perform a basic fuzzy matching check for inconsistencies in object (string) columns.
        inconsistency_issues = {}
        for col in df.select_dtypes(include=['object']).columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 1: # Only check if there's more than one unique value
                # Check for near duplicates among unique values using fuzzy matching
                for i in range(len(unique_values)):
                    for j in range(i + 1, len(unique_values)):
                        val1 = str(unique_values[i])
                        val2 = str(unique_values[j])
                        if fuzz.ratio(val1, val2) > 85: # High similarity threshold (e.g., 85% match)
                            if col not in inconsistency_issues:
                                inconsistency_issues[col] = []
                            inconsistency_issues[col].append(f"'{val1}' and '{val2}' are highly similar.")

        if inconsistency_issues:
            validation_report["status"] = "failed"
            validation_report["issues_found"]["inconsistencies"] = inconsistency_issues
            validation_report["summary"] = "Cleaning validation failed: Inconsistencies still present."
            validation_success = False
            self.logger.warning(f"Validation Issue: Inconsistencies found after cleaning: {inconsistency_issues}")
            
            # Generate recommendations for handling inconsistencies
            for col, details_list in inconsistency_issues.items():
                if col not in recommendations["cleaning_operations"]:
                    recommendations["cleaning_operations"][col] = []
                recommendations["cleaning_operations"][col].append({
                    "operation": "handle_inconsistencies",
                    "reason": f"Still inconsistencies found. Re-run inconsistency handling."
                })

        self.logger.info(f"Cleaning validation finished with status: {validation_report['status']}")
        return validation_report, validation_success, recommendations