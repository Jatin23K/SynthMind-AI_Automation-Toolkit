# api.py
# This file defines the FastAPI application for the Meta Minds data analysis workflow.
# It provides an endpoint to trigger the analysis process by accepting file paths.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

# Import necessary functions from your modules
from data_loader import read_file
from data_analyzer import generate_summary
from main import create_agents, create_tasks, run_crew_standard, format_output_final, create_comparison_task # Assuming these are needed directly or helper functions are moved/adapted

# Configure logging (optional, if not already done in config.py)
logging.basicConfig(level=logging.INFO)

# Initialize the FastAPI application
app = FastAPI()

# Define the request body schema using Pydantic
class FilePathsRequest(BaseModel):
    file_paths: List[str]
    """List of file paths (strings) to the datasets to be analyzed."""

# Define the root endpoint
@app.get("/")
def read_root():
    """Root endpoint returning a simple greeting."""
    return {"Hello": "World"}

# Define the analysis endpoint
@app.post("/analyze")
async def analyze_datasets(request: FilePathsRequest):
    """Analyzes datasets provided via file paths using the CrewAI workflow.

    Args:
        request (FilePathsRequest): A Pydantic model containing a list of file paths.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis output.

    Raises:
        HTTPException: If no valid datasets can be loaded or if there's an error
                       during agent/task creation or CrewAI execution.
    """
    logging.info(f"Received analysis request for files: {request.file_paths}")
    datasets = []
    dataset_summaries = {}
    task_results = []
    task_headers = []

    # 1. Load and process datasets from the provided file paths
    loaded_datasets = []
    for file_path in request.file_paths:
        try:
            # Use the read_file function from data_loader.py
            df = read_file(file_path)
            # Extract dataset name from the file path
            dataset_name = file_path.split('/')[-1] # Simple name extraction
            loaded_datasets.append((dataset_name, df))
            logging.info(f"Successfully loaded {dataset_name}")
        except (FileNotFoundError, ValueError) as e:
            # Log specific errors for file loading issues
            logging.error(f"Error loading file {file_path}: {e}")
            # Skip the file and log the error, continue with others
            continue
        except Exception as e:
             # Catch any other unexpected errors during reading
             logging.error(f"Unexpected error loading file {file_path}: {e}")
             continue

    # Raise an error if no datasets were successfully loaded
    if not loaded_datasets:
        raise HTTPException(status_code=400, detail="No valid datasets could be loaded.")

    # 2. Generate summaries for each loaded dataset
    for name, df in loaded_datasets:
        try:
            # Use the generate_summary function from data_analyzer.py
            summary = generate_summary(df)
            dataset_summaries[name] = summary
            logging.info(f"Generated summary for {name}")
        except Exception as e:
            # Log errors during summary generation
            logging.error(f"Error generating summary for {name}: {e}")
            dataset_summaries[name] = {"error": str(e)} # Store error in summary

    # 3. Create agents and tasks for the CrewAI workflow
    try:
        # Create agents using the create_agents function
        agents = create_agents()
        
        # Initialize lists for all tasks and their headers
        all_tasks = []
        all_task_headers = []

        # Create individual analysis tasks for each dataset
        # Assumes create_tasks takes the list of (name, df) tuples and agents
        individual_tasks_info = create_tasks(loaded_datasets, agents[0], agents[1]) # Pass agents explicitly
        all_tasks.extend(individual_tasks_info['tasks'])
        all_task_headers.extend(individual_tasks_info['headers'])

        # Create comparison task if more than one dataset is provided
        if len(loaded_datasets) > 1:
             # Assumes create_comparison_task takes the list of (name, df) tuples and an agent
             comparison_task_info = create_comparison_task(loaded_datasets, agents[1]) # Pass question_genius agent
             # Check if a comparison task was actually created (function returns None if not needed)
             if comparison_task_info:
                  all_tasks.append(comparison_task_info['task'])
                  all_task_headers.append(comparison_task_info['header'])

        logging.info(f"Created {len(all_tasks)} tasks.")

    except Exception as e:
        # Log and raise an error if agent or task creation fails
        logging.error(f"Error creating agents or tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating analysis tasks: {e}")

    # 4. Run CrewAI tasks if any tasks were created
    if all_tasks:
        try:
            # Run the tasks using the run_crew_standard function
            task_results = run_crew_standard(all_tasks, agents)
            logging.info("CrewAI tasks finished running.")
        except Exception as e:
            # Log errors during CrewAI execution
            logging.error(f"Error running CrewAI tasks: {e}")
            # Populate task_results with error messages for all tasks if execution fails
            task_results = [f"Error during CrewAI execution: {e}"] * len(all_tasks) 
    else:
        logging.warning("No tasks to run.")
        task_results = [] # Ensure task_results is an empty list if no tasks

    # 5. Format the final output using summaries and task results
    try:
        # Use the format_output_final function from output_handler.py
        final_output_lines = format_output_final(dataset_summaries, task_results, all_task_headers)
        logging.info("Output formatted.")
    except Exception as e:
        # Log errors during output formatting
        logging.error(f"Error formatting final output: {e}")
        # Fallback formatting in case of error
        final_output_lines = ["Error formatting output.", str(e), "--- Raw Summaries ---"]
        for name, summary in dataset_summaries.items():
             final_output_lines.append(f"{name}: {summary}")
        final_output_lines.append("--- Raw Task Results ---")
        final_output_lines.extend(task_results)

    # 6. Return the final analysis output
    # The output is returned as a dictionary containing a list of strings
    return {"analysis_output": final_output_lines}

# Note: The functions imported from main.py (create_agents, create_tasks, run_crew_standard, format_output_final)
# are assumed to be adapted to accept DataFrames directly instead of relying on user input or file paths within them.
# Ensure necessary imports (like pandas) are present in api.py or the imported modules if needed directly.
