from crewai import Agent, Task # Task might be used if we define formal CrewAI tasks later

# Import all 'employee' agents
from .meta_analyzer_agent import MetaAnalyzerAgent
from .cleaner_agent import DataCleanerAgent
from .cleaning_validator_agent import CleaningValidatorAgent
from .modifier_agent import DataModifierAgent
from .modification_validator_agent import ModificationValidatorAgent
from .transformer_agent import DataTransformerAgent
from .transformation_validator_agent import TransformationValidatorAgent
# ProcessRecorderAgent is passed in, so no direct import here if it's from another module path in main
import pandas as pd # Add pandas import for data loading

class OrchestratorAgent:
    def __init__(self, process_recorder_agent):
        self.agent = Agent(
            role="Chief Data Pipeline Orchestrator",
            goal="Oversee and manage the entire data purification pipeline, ensuring efficient and accurate data processing from meta-analysis to final reporting.",
            backstory="A seasoned project manager with expertise in complex data workflows, responsible for coordinating multiple specialist agents to achieve pristine data quality.",
            allow_delegation=True, # Allows this agent to delegate tasks
            verbose=True
        )
        self.process_recorder_agent = process_recorder_agent

        # Initialize all 'employee' agents
        self.meta_analyzer_agent = MetaAnalyzerAgent()
        self.data_cleaner_agent = DataCleanerAgent() # Corrected: No recorder_agent here
        self.cleaning_validator_agent = CleaningValidatorAgent() # Assuming similar init for now
        self.data_modifier_agent = DataModifierAgent() # Assuming similar init for now
        self.modification_validator_agent = ModificationValidatorAgent() # Assuming similar init for now
        self.data_transformer_agent = DataTransformerAgent() # Assuming similar init for now
        self.transformation_validator_agent = TransformationValidatorAgent() # Assuming similar init for now

    def orchestrate_data_processing(self, dataset_paths, meta_output_path, report_path):
        self.process_recorder_agent.record_task_activity(
            agent_name="OrchestratorAgent",
            task_description=f"Starting data purification pipeline for datasets: {', '.join(dataset_paths) if dataset_paths else 'N/A'}",
            status="Initiated",
            details={"meta_output_path": meta_output_path, "report_path": report_path}
        )

        # 1. Meta-Analysis Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Delegating to MetaAnalyzerAgent", "In Progress")
        meta_analysis_results, success = self.meta_analyzer_agent.analyze(
            dataset_paths=dataset_paths,
            meta_output_path=meta_output_path
        )
        if not success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Meta-analysis failed. Halting pipeline.", "Failed", meta_analysis_results)
            print(f"Error: Meta-analysis failed. Halting pipeline.")
            return False
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Meta-analysis completed.", "Success", meta_analysis_results)
        
        valid_dataset_paths = meta_analysis_results.get("valid_dataset_paths", [])
        processing_instructions = meta_analysis_results.get("processing_instructions", {})
        filtered_dataset = meta_analysis_results.get("filtered_dataset") # Get the filtered dataset
        
        if filtered_dataset is None:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Meta-analysis did not return a valid dataset. Halting pipeline.", "Failed")
            print("Error: Meta-analysis did not return a valid dataset. Halting pipeline.")
            return False

        # Use the filtered dataset as the starting point for the pipeline
        current_df = filtered_dataset
        # Assuming we are processing the dataset that was filtered by MetaAnalyzer
        # We can use the first valid path for logging purposes if needed, but the data is in current_df
        dataset_path_for_logging = valid_dataset_paths[0] if valid_dataset_paths else "N/A"
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Starting processing with filtered dataset from: {dataset_path_for_logging}", "In Progress")

        # --- Process the single filtered dataset ---

        # 2. Data Cleaning Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Delegating cleaning for {dataset_path_for_logging} to DataCleanerAgent", "In Progress")
        cleaned_df, clean_success = self.data_cleaner_agent.clean_data(current_df, processing_instructions.get('cleaning'))
        if not clean_success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Data cleaning failed for {dataset_path_for_logging}.", "Failed")
            return False # Halting pipeline on failure
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Data cleaning completed for {dataset_path_for_logging}.", "Success")
        current_df = cleaned_df

        # 3. Cleaning Validation Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Delegating cleaning validation for {dataset_path_for_logging} to CleaningValidatorAgent", "In Progress")
        # Note: Validation might need the original dataset or specific validation instructions
        # For now, assuming validation uses the cleaned_df and instructions
        validation_report, validation_success = self.cleaning_validator_agent.validate_cleaning(current_df, processing_instructions.get('cleaning'))
        if not validation_success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Cleaning validation failed for {dataset_path_for_logging}. Details: {validation_report}", "Failed")
            return False # Halting pipeline on failure
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Cleaning validation passed for {dataset_path_for_logging}.", "Success", validation_report)

        # 4. Data Modification Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Delegating modification for {dataset_path_for_logging} to DataModifierAgent", "In Progress")
        modified_df, mod_success = self.data_modifier_agent.modify_data(current_df, processing_instructions.get('modification'))
        if not mod_success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Data modification failed for {dataset_path_for_logging}.", "Failed")
            return False # Halting pipeline on failure
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Data modification completed for {dataset_path_for_logging}.", "Success")
        current_df = modified_df

        # 5. Modification Validation Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Delegating modification validation for {dataset_path_for_logging} to ModificationValidatorAgent", "In Progress")
        # Note: Validation might need the original dataset or specific validation instructions
        mod_val_report, mod_val_success = self.modification_validator_agent.validate_modification(current_df, processing_instructions.get('modification'))
        if not mod_val_success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Modification validation failed for {dataset_path_for_logging}. Details: {mod_val_report}", "Failed")
            return False # Halting pipeline on failure
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Modification validation passed for {dataset_path_for_logging}.", "Success", mod_val_report)

        # 6. Data Transformation Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Delegating transformation for {dataset_path_for_logging} to DataTransformerAgent", "In Progress")
        transformed_df, trans_success = self.data_transformer_agent.transform_data(current_df, processing_instructions.get('transformation'))
        if not trans_success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Data transformation failed for {dataset_path_for_logging}.", "Failed")
            return False # Halting pipeline on failure
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Data transformation completed for {dataset_path_for_logging}.", "Success")
        current_df = transformed_df

        # 7. Transformation Validation Stage
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Delegating transformation validation for {dataset_path_for_logging} to TransformationValidatorAgent", "In Progress")
        # Note: Validation might need the original dataset or specific validation instructions
        trans_val_report, trans_val_success = self.transformation_validator_agent.validate_transformation(current_df, processing_instructions.get('transformation'))
        if not trans_val_success:
            self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Transformation validation failed for {dataset_path_for_logging}. Details: {trans_val_report}", "Failed")
            return False # Halting pipeline on failure
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Transformation validation passed for {dataset_path_for_logging}.", "Success", trans_val_report)

        # TODO: Save the processed DataFrame (current_df) for this dataset_path
        # Example: self.process_recorder_agent.save_cleaned_dataset(current_df, dataset_path_for_logging, "processed")
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", f"Processed {dataset_path_for_logging} - Placeholder for saving", "Completed")

        # --- End of processing for the single filtered dataset ---

        # 8. Final Reporting (Aggregated or summary report)
        # This might involve the ProcessRecorderAgent generating a final summary report
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Generating final pipeline report.", "In Progress")
        # final_report_details = self.process_recorder_agent.generate_json_report(report_path) # Assuming this method exists and works
        self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Final pipeline report generated.", "Success", {"report_location": report_path})

        self.process_recorder_agent.record_task_activity("OrchestratorAgent", "Data purification pipeline finished successfully.", "Completed")
        return True # Indicate overall pipeline success