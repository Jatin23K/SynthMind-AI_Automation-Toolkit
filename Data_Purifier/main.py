import os
import pandas as pd
import sys
from data_purifier.agents.meta_analyzer_agent import MetaAnalyzerAgent
from data_purifier.agents.cleaner_agent import DataCleanerAgent
from data_purifier.agents.cleaning_validator_agent import CleaningValidatorAgent
from data_purifier.agents.modifier_agent import DataModifierAgent
from data_purifier.agents.modification_validator_agent import ModificationValidatorAgent
from data_purifier.agents.transformer_agent import DataTransformerAgent
from data_purifier.agents.transformation_validator_agent import TransformationValidatorAgent
from data_purifier.agents.process_recorder_agent import ProcessRecorderAgent
from data_purifier.utils.file_utils import get_input_files
from data_purifier.config.settings import load_config
from data_purifier.tasks.meta_analyzer_tasks import MetaAnalyzerTasks
from data_purifier.tasks.cleaner_tasks import CleanerTasks
from crewai import Crew, Process


def run_purification_pipeline(dataset_paths: list[str], meta_output_path: str):
    # Load API Key
    load_config()
    
    # Initialize agents
    process_recorder = ProcessRecorderAgent()
    meta_analyzer = MetaAnalyzerAgent()
    cleaner = DataCleanerAgent()
    cleaning_validator = CleaningValidatorAgent(recorder_agent=process_recorder)
    modifier = DataModifierAgent()
    modification_validator = ModificationValidatorAgent(recorder_agent=process_recorder)
    transformer = DataTransformerAgent()
    transformation_validator = TransformationValidatorAgent()
    
    # --- Execute Meta Analysis ---
    # Call meta_analyzer directly
    analysis_result, analysis_success = meta_analyzer.analyze(dataset_paths, meta_output_path)
    
    if not analysis_success:
        print("Meta analysis failed. Exiting.")
        return {"status": "failed", "message": "Meta analysis failed."}
        
    # Update dataset paths with validated paths and get instructions
    valid_dataset_paths = analysis_result["valid_dataset_paths"]
    processing_instructions = analysis_result["processing_instructions"]
    
    if not valid_dataset_paths:
        print("No valid dataset paths found after meta analysis. Exiting.")
        return {"status": "failed", "message": "No valid dataset paths found after meta analysis."}

    processed_files = []
    reports = []
    
    # Process each dataset
    for i, dataset_path in enumerate(valid_dataset_paths):
        try:
            print(f"\n--- Processing Dataset {i+1}/{len(valid_dataset_paths)}: {dataset_path} ---")
            # Load dataset
            df = pd.read_csv(dataset_path)
            original_df = df.copy() # Keep original for validation if needed
            
            # Step 1: Cleaning
            print("Starting Cleaning Process...")
            # Call cleaner.process directly with required arguments
            df_cleaned, cleaning_success, cleaning_summary = cleaner.process(
                df_input=df, 
                processing_instructions=processing_instructions.get('cleaning', {}), # Pass relevant instructions
                recorder_agent=process_recorder # Pass the recorder agent
            )
            
            if not cleaning_success:
                print(f"Cleaning failed for dataset {dataset_path}. Summary: {cleaning_summary}")
                # Decide whether to continue or skip this dataset
                process_recorder.record_task_activity(
                    agent_name="Cleaner",
                    task_description=f"Cleaning failed for dataset {dataset_path}",
                    status="Failed",
                    details={"summary": cleaning_summary}
                )
                continue # Skip to the next dataset on failure
            
            print(f"Cleaning completed. Summary: {cleaning_summary}")
            process_recorder.record_task_activity(
                agent_name="Cleaner",
                task_description=f"Cleaning completed for dataset {dataset_path}",
                status="Success",
                details={"summary": cleaning_summary}
            )

            # Step 1.1: Validate Cleaning
            print("Validating Cleaning Results...")
            # Call cleaning_validator.validate directly
            is_cleaning_valid = cleaning_validator.validate(df_cleaned, df)
            
            if not is_cleaning_valid:
                print(f"Cleaning validation failed for dataset {dataset_path}.")
                # Log validation failure details if available in validator agent
                process_recorder.record_task_activity(
                    agent_name="Cleaning Validator",
                    task_description=f"Cleaning validation failed for dataset {dataset_path}",
                    status="Failed"
                    # Add more details if available from validator agent
                )
                continue # Skip to the next dataset on validation failure
            
            print("Cleaning validation successful.")
            process_recorder.record_task_activity(
                agent_name="Cleaning Validator",
                task_description=f"Cleaning validation successful for dataset {dataset_path}",
                status="Success"
            )
            
            # Step 2: Modification
            print("Starting Modification Process...")
            # Assuming modifier.modify has a signature like modify(self, df)
            # If it needs instructions or recorder, pass them similarly
            # Based on modifier_agent.py snippet, it has modify(self, df)
            # The main.py snippet calls modifier.process, which is another mismatch.
            # Let's assume the method to call is modify based on the agent class.
            df_modified, modification_success = modifier.modify(df_cleaned) # Call modify, not process
            
            if not modification_success:
                print(f"Modification failed for dataset {dataset_path}.")
                process_recorder.record_task_activity(
                    agent_name="Modifier",
                    task_description=f"Modification failed for dataset {dataset_path}",
                    status="Failed"
                    # Add more details if available from modifier agent
                )
                continue # Skip to the next dataset on failure
            
            print("Modification completed.")
            process_recorder.record_task_activity(
                agent_name="Modifier",
                task_description=f"Modification completed for dataset {dataset_path}",
                status="Success"
            )

            # Step 2.1: Validate Modification
            print("Validating Modification Results...")
            # Pass the state before modification for validation.
            is_modification_valid = modification_validator.validate(df_modified, df_cleaned) 
            
            if not is_modification_valid:
                print(f"Modification validation failed for dataset {dataset_path}.")
                process_recorder.record_task_activity(
                    agent_name="Modification Validator",
                    task_description=f"Modification validation failed for dataset {dataset_path}",
                    status="Failed"
                    # Add more details if available from validator agent
                )
                continue # Skip to the next dataset on validation failure
            
            print("Modification validation successful.")
            process_recorder.record_task_activity(
                agent_name="Modification Validator",
                task_description=f"Modification validation successful for dataset {dataset_path}",
                status="Success"
            )
            
            # Step 3: Transformation
            print("Starting Transformation Process...")
            # Assuming transformer.process has a signature like cleaner.process:
            # process(self, df_input, processing_instructions, recorder_agent)
            # *** NOTE: Verify TransformerAgent.process signature ***
            df_transformed, transformation_success, transformation_summary = transformer.process(
                 df_input=df_modified,
                 processing_instructions=processing_instructions.get('transformation', {}), # Pass relevant instructions
                 recorder_agent=process_recorder # Pass the recorder agent
            )

            if not transformation_success:
                print(f"Transformation failed for dataset {dataset_path}. Summary: {transformation_summary}")
                process_recorder.record_task_activity(
                    agent_name="Transformer",
                    task_description=f"Transformation failed for dataset {dataset_path}",
                    status="Failed",
                    details={"summary": transformation_summary}
                )
                continue # Skip to the next dataset on failure
            
            print(f"Transformation completed. Summary: {transformation_summary}")
            process_recorder.record_task_activity(
                agent_name="Transformer",
                task_description=f"Transformation completed for dataset {dataset_path}",
                status="Success",
                details={"summary": transformation_summary}
            )

            # Step 3.1: Validate Transformation
            print("Validating Transformation Results...")
            # Assuming transformation_validator.validate has a signature like validate(self, df, original_df)
            # Similar to modification validation, this might need the state *before* transformation.
            # Pass the state before transformation for validation.
            is_transformation_valid = transformation_validator.validate(df_transformed, df_modified)
            
            if not is_transformation_valid:
                print(f"Transformation validation failed for dataset {dataset_path}.")
                process_recorder.record_task_activity(
                    agent_name="Transformation Validator",
                    task_description=f"Transformation validation failed for dataset {dataset_path}",
                    status="Failed"
                    # Add more details if available from validator agent
                )
                continue # Skip to the next dataset on validation failure
            
            print("Transformation validation successful.")
            process_recorder.record_task_activity(
                agent_name="Transformation Validator",
                task_description=f"Transformation validation successful for dataset {dataset_path}",
                status="Success"
            )

            # Step 4: Finalize Reports and Save Dataset
            print("Finalizing Reports and Saving Dataset...")
            # The process_recorder.record_process method signature in main.py is:
            # record_process(self, meta_logs, cleaning_logs, modification_logs, transformation_logs)
            # You need to collect logs from each agent.
            # Assuming agents store logs in attributes like .analysis_logs, .cleaning_logs, etc.
            # *** NOTE: Verify agent log attribute names ***
            final_report_content = process_recorder.record_process(
                getattr(meta_analyzer, 'analysis_logs', {}), # Use getattr with default empty dict
                getattr(cleaner, 'cleaning_logs', {}), 
                getattr(modifier, 'modification_logs', {}), 
                getattr(transformer, 'transformation_logs', {})
            )
            
            # Define output paths
            base_name = os.path.basename(dataset_path)
            name, ext = os.path.splitext(base_name)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
            processed_output_filename = f"{name}_processed_{timestamp}{ext}"
            report_output_filename = f"{name}_report_{timestamp}.md"
            
            # Assuming you want to save processed files and reports in the current directory or a specified report_path
            # Using current directory for simplicity based on previous snippet
            processed_output_path = processed_output_filename
            report_output_path = report_output_filename

            # Save processed dataset
            try:
                df_transformed.to_csv(processed_output_path, index=False)
                print(f"Processed dataset saved to {processed_output_path}")
                processed_files.append(processed_output_path)
            except Exception as e:
                print(f"Error saving processed dataset {processed_output_path}: {str(e)}")
                process_recorder.record_task_activity(
                    agent_name="main_loop",
                    task_description=f"Error saving processed dataset {processed_output_path}",
                    status="Failed",
                    details={"error": str(e)}
                )

            # Save process report
            try:
                with open(report_output_path, "w") as f:
                    f.write(final_report_content)
                print(f"Process report saved to {report_output_path}")
                reports.append(report_output_path)
            except Exception as e:
                print(f"Error saving process report {report_output_path}: {str(e)}")
                process_recorder.record_task_activity(
                    agent_name="main_loop",
                    task_description=f"Error saving process report {report_output_path}",
                    status="Failed",
                    details={"error": str(e)}
                )
                
            print(f"--- Finished Processing Dataset {dataset_path} ---")
            
        except Exception as e:
            print(f"An unexpected error occurred while processing dataset {dataset_path}: {str(e)}")
            # Log the error using the recorder agent if possible
            process_recorder.record_task_activity(
                agent_name="main_loop",
                task_description=f"Unexpected error processing dataset {dataset_path}",
                status="Failed",
                details={"error": str(e)}
            )
            continue # Continue to the next dataset even if one fails

    print("\n--- Data Purification Pipeline Finished ---")
    # You might want to generate a final summary report for the entire run here
    # using the process_recorder_agent's aggregated logs.
    # Example: process_recorder.generate_final_run_report("final_pipeline_summary.md")
    
    return {"status": "success", "processed_files": processed_files, "reports": reports}


def main():
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Parse command line arguments according to usage pattern:
        # python main.py <input_file1,input_file2,...> <meta_output_path>
        if len(sys.argv) < 3:
            print("Usage: python main.py <input_file1,input_file2,...> <meta_output_path>")
            sys.exit(1)
            
        # Parse the first argument as a comma-separated list of input files
        dataset_paths = sys.argv[1].split(',')
        meta_output_path = sys.argv[2]
        
        print(f"Using command-line arguments:")
        print(f"Dataset paths: {dataset_paths}")
        print(f"Meta output path: {meta_output_path}")
    else:
        # If no command-line arguments are provided, prompt the user for input
        print("No command-line arguments provided. Entering interactive mode.")
        
        while True:
            try:
                num_datasets_input = input("Enter the number of datasets: ")
                num_datasets = int(num_datasets_input)
                if num_datasets <= 0:
                    print("Please enter a positive integer.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        dataset_paths = []
        for i in range(num_datasets):
            while True:
                dataset_path = input(f"Enter path for dataset {i+1}: ")
                if os.path.exists(dataset_path):
                    dataset_paths.append(dataset_path)
                    break
                else:
                    print(f"Error: File not found at '{dataset_path}'. Please enter a valid path.")

        while True:
            meta_output_path = input("Enter path for meta output file: ")
            # Basic validation: check if parent directory exists or can be created
            output_dir = os.path.dirname(meta_output_path) or '.'
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    break
                except OSError as e:
                    print(f"Error creating directory '{output_dir}': {e}")
            else:
                 break # Directory exists, proceed

        print(f"Using interactive input:")
        print(f"Dataset paths: {dataset_paths}")
        print(f"Meta output path: {meta_output_path}")
    
    # Run the purification pipeline with the provided inputs
    result = run_purification_pipeline(dataset_paths, meta_output_path)
    
    # Print the result status
    if result["status"] == "success":
        print("\nData purification completed successfully!")
        print(f"Processed files: {result['processed_files']}")
        print(f"Reports: {result['reports']}")
    else:
        print(f"\nData purification failed: {result['message']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Usage: python main.py <input_file1,input_file2,...> <meta_output_path>")
        sys.exit(1)
