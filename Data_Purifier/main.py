# main.py
# This is the main entry point for the Data Purification System.
# It orchestrates the entire data processing pipeline, from input to final output.

import logging
import time
import pandas as pd
import os
import argparse # Import argparse for command-line argument parsing
import sys # Import sys for system-specific parameters and functions

# Import necessary agents from their respective modules
from agents.orchestrator_agent import OrchestratorAgent
from agents.process_recorder_agent import ProcessRecorderAgent
from agents.meta_analyzer_agent import MetaAnalyzerAgent
from config.settings import load_config # Configuration loader

def setup_logging(verbose: bool):
    """
    Sets up the logging configuration for the application.
    Logs are directed to both a file ('data_purification.log') and the console.

    Args:
        verbose (bool): If True, sets the logging level to DEBUG for more detailed output.
                        Otherwise, sets it to INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_purification.log'), # Logs to a file
            logging.StreamHandler() # Logs to the console
        ]
    )
    return logging.getLogger(__name__)

def main():
    """
    Main entry point for the data purification system.
    It handles command-line arguments for file paths, initializes agents, and
    orchestrates the data processing pipeline.
    """
    print("--- Data Purification System ---")

    # --- Command-line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Data Purification System.")
    parser.add_argument('--dataset_paths', nargs='+', required=True,
                        help='Absolute path(s) to the input dataset(s) (e.g., C:\\data\\my_data.csv).')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Absolute path to save the processed dataset (e.g., C:\\output\\cleaned_data.csv).')
    parser.add_argument('--meta_output_path', type=str, required=True,
                        help='Absolute path to the meta output file (e.g., C:\\config\\meta.txt).')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level).')

    args = parser.parse_args()

    dataset_paths = args.dataset_paths
    output_path = args.output_path
    meta_output_path = args.meta_output_path
    verbose = args.verbose

    # Load configuration settings from config/settings.py and .env file
    try:
        config = load_config()
    except EnvironmentError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure all required environment variables are set.")
        sys.exit(1) # Use sys.exit for clean exit in scripts

    # Set up logging based on user preference and configuration
    logger = setup_logging(verbose or config.get("log_level", "INFO").upper() == "DEBUG")
    logger.info(f"Starting data purification for datasets: {dataset_paths}")

    # --- Agent Initialization ---
    # Initialize the core agents responsible for different parts of the pipeline.

    # ProcessRecorderAgent: Records all activities and generates a final report.
    process_recorder_agent = ProcessRecorderAgent()

    # OrchestratorAgent: Manages the overall flow, delegates tasks, and handles feedback loops.
    orchestrator = OrchestratorAgent(process_recorder_agent=process_recorder_agent, config=config)

    # MetaAnalyzerAgent: Analyzes metadata, loads raw data, and prepares it for processing.
    # This agent also handles the internal conversion of CSV to Feather for efficiency.
    meta_analyzer_agent = MetaAnalyzerAgent(config=config)

    # --- Data Loading and Initial Analysis ---
    # The MetaAnalyzerAgent is responsible for loading the raw datasets,
    # performing initial schema analysis, and converting them to an efficient
    # internal format (Feather) for subsequent processing.
    try:
        logger.info("Loading and analyzing datasets using MetaAnalyzerAgent...")
        df, meta_analysis_report = meta_analyzer_agent.analyze(
            num_datasets=len(dataset_paths), # num_datasets is now derived from len(dataset_paths)
            dataset_paths=dataset_paths,
            meta_output_path=meta_output_path
        )
        logger.info(f"Datasets loaded and initial meta-analysis complete. DataFrame shape: {df.shape}")
        # The meta_analysis_report contains the pipeline plan and suggested operations
        # which the OrchestratorAgent will use.
        orchestrator.set_processing_instructions(meta_analysis_report)

    except Exception as e:
        logger.error(f"Critical error during meta-analysis or data loading: {e}", exc_info=True)
        print(f"Data purification failed at meta-analysis stage. Check logs for details.")
        sys.exit(1) # Use sys.exit for clean exit in scripts

    # --- Data Purification Pipeline Execution ---
    # The OrchestratorAgent takes the prepared DataFrame and executes the
    # cleaning, modification, and transformation stages.
    start_time = time.time()
    logger.info("Starting orchestrated data processing pipeline...")
    success, cleaned_df = orchestrator.orchestrate_data_processing(
        df=df,
        meta_output_path=meta_output_path, # Passed for context, though primary use is in MetaAnalyzer
        report_path=output_path # Path where the final processed CSV will be saved
    )
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

    # --- Final Output and Reporting ---
    if success:
        logger.info("Data purification completed successfully!")
        print("\nData purification completed successfully!")
        print(f"Processed dataset saved to: {output_path}")
        # The final report is generated by the ProcessRecorderAgent via the Orchestrator
    else:
        logger.error("Data purification failed. Check the logs and 'data_purification.log' for details.")
        print("\nData purification failed. Check the logs for details.")
        sys.exit(1) # Use sys.exit for clean exit in scripts

if __name__ == '__main__':
    main()
