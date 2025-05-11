# =========================================================
# main.py: Meta Minds Application Entry Point and Orchestrator
# =========================================================
# This script orchestrates the entire workflow:
# 1. Gets user input for dataset paths.
# 2. Loads and processes the datasets.
# 3. Generates data summaries and column descriptions using GPT.
# 4. Creates CrewAI agents and tasks based on the data.
# 5. Runs the CrewAI tasks to generate analytical questions.
# 6. Formats the collected summaries and questions.
# 7. Saves the final output to a file.
# Imports are handled centrally in config.py where appropriate (like the OpenAI client).
# Logging is also configured in config.py.

import os
import logging
import pandas as pd # Needed here for pd.DataFrame type hints and potentially for df operations

# Import modules for different parts of the workflow
# Note: The OpenAI client and basic logging are configured in config.py
from data_loader import read_file
from data_analyzer import generate_summary # generate_summary uses the client from config
from agents import create_agents
from tasks import create_tasks, create_comparison_task
from output_handler import save_output

# CrewAI components needed for orchestration within main.py
from crewai import Crew, Process, Agent, Task # Explicitly import necessary CrewAI objects

# --- Helper Functions (Included here as part of the orchestration layer) ---
# These functions define the specific steps and logic within the main workflow.
# In a much larger application, these might be moved to a dedicated 'workflow_runner.py' module.

def get_user_input_file_paths() -> list[str]:
    """Prompts the user for the number of datasets and their file paths."""
    file_paths = []
    try:
        # Use a loop to robustly get the number of files
        while True:
            num_files_str = input("Enter number of datasets you want to analyze (e.g., 1, 2): ").strip()
            try:
                num_files = int(num_files_str)
                if num_files >= 0: # Allow 0 files, handled later
                    break # Valid input, exit loop
                else:
                    logging.warning("Number of datasets cannot be negative. Please enter a non-negative number.")
            except ValueError:
                logging.warning(f"Invalid input: '{num_files_str}'. Please enter a number.")

        if num_files == 0:
            logging.info("User entered 0 datasets.")
            return [] # Return empty list

        logging.info(f"Expecting {num_files} dataset path(s).")
        for i in range(num_files):
            while True: # Loop until a non-empty path is entered for each file
                file_path = input(f"Enter full path of dataset {i+1} (CSV, XLSX, or JSON): ").strip()
                if file_path:
                    file_paths.append(file_path)
                    break
                else:
                    logging.warning("File path cannot be empty. Please enter a valid path.")

    except EOFError:
         logging.error("Input stream closed unexpectedly while waiting for user input.")
         return []
    except Exception as e:
         logging.error(f"An unexpected error occurred during user input: {e}")
         return []

    return file_paths


def process_datasets(file_paths: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """Loads datasets from provided file paths using the data_loader module."""
    datasets = []
    if not file_paths:
        logging.warning("No file paths provided to process_datasets.")
        return []

    logging.info("Starting dataset processing...")
    for file_path in file_paths:
        try:
            df = read_file(file_path) # Uses the read_file function from data_loader.py
            dataset_name = os.path.basename(file_path)
            datasets.append((dataset_name, df))
            logging.info(f"Successfully loaded dataset: {dataset_name} (Shape: {df.shape})")
            if df.empty:
                 logging.warning(f"Dataset '{dataset_name}' is empty.")

        except FileNotFoundError:
             logging.error(f"Skipping {file_path}: File not found.")
        except ValueError as ve:
             logging.error(f"Skipping {file_path}: {ve}") # Log unsupported file type errors etc.
        except Exception as e:
            # Catch any other unexpected errors during reading
            logging.error(f"Skipping {file_path} due to unexpected error during load: {e}")
            continue # Skip this file and try the next one

    if not datasets:
         logging.error("No valid datasets could be loaded from the provided paths.")
    else:
         logging.info(f"Finished processing. Successfully loaded {len(datasets)} dataset(s).")

    return datasets

# --- REVISED run_crew_standard function ---
# Runs each task in a separate Crew instance sequentially.
# This aligns with the original code's apparent intent of independent task execution
# and result reporting per task/comparison.
def run_crew_standard(tasks: list[Task], agents: list[Agent]) -> list[str]:
     """Runs the CrewAI process by executing tasks sequentially in separate Crews."""
     if not tasks:
          logging.warning("No tasks provided to run_crew_standard. Skipping execution.")
          return [] # Return empty list if no tasks

     logging.info("🚀 Starting CrewAI task execution...")
     task_results = []
     # Provide the full list of possible agents to each single-task Crew
     all_agents_roster = list(set(agents)) # Ensure unique agent instances in the roster

     for i, task in enumerate(tasks):
         # Use the task's expected_output as a way to identify the task in logs/results
         task_identifier = task.expected_output if task.expected_output else f"Task {i+1}"
         logging.info(f"--- Running {task_identifier} ---")

         try:
             # Create a new Crew for THIS specific task
             crew = Crew(
                 agents=all_agents_roster, # Provide the full roster of agents to the crew
                 tasks=[task],            # The crew will only execute this single task
                 process=Process.sequential, # Even with one task, sequential is a valid process
                 verbose=True              # Show detailed agent steps
             )
             # kickoff() with Process.sequential for a single task returns the result of that task
             result = crew.kickoff()
             task_results.append(str(result)) # Store result (CrewAI output is often a string)
             logging.info(f"--- Finished {task_identifier} ---")
         except Exception as e:
             logging.error(f"An error occurred running {task_identifier}: {e}")
             task_results.append(f"Error executing task '{task_identifier}': {e}") # Store error message

     logging.info("✅ All CrewAI tasks finished execution attempt.")
     return task_results # Return list of results/errors from each task kickoff

# --- REVISED format_output_final function ---
# Takes pre-generated summaries and task results to format the final output string list.
def format_output_final(dataset_summaries: dict, task_results: list[str], task_headers: list[str]) -> list[str]:
     """Formats data summaries (pre-generated) and task results into a list of lines for output."""
     logging.info("Formatting output...")
     output_lines = []

     # Add Data Summaries
     logging.info("Adding data summaries to output.")
     # Iterate through dataset_summaries dictionary. Order might not be guaranteed
     # unless using an OrderedDict or preserving names in a list.
     # Assuming keys are dataset names as used elsewhere.
     for name, summary in dataset_summaries.items():
         output_lines.append(f"====== DATA SUMMARY FOR {name} ======")
         if "error" in summary:
              output_lines.append(f"Error generating summary: {summary['error']}")
         else:
             # Using .get() for safe access in case structure is unexpected
             output_lines.append(f"Rows: {summary.get('rows', 'N/A')}")
             output_lines.append(f"Columns: {summary.get('columns', 'N/A')}")
             column_info = summary.get('column_info')
             if column_info and isinstance(column_info, dict):
                 for col, info in column_info.items():
                     if isinstance(info, dict):
                          output_lines.append(f"{col} ({info.get('dtype', 'N/A')}): {info.get('description', 'Description unavailable')}")
                     else:
                          output_lines.append(f"{col} (Info structure error for column)")
             else:
                  output_lines.append("Summary column info unavailable or malformed.")
         output_lines.append("") # Add blank line after each summary

     # Add Generated Questions from Task Results
     output_lines.append("====== GENERATED QUESTIONS ======")
     logging.info("Adding generated questions from task results.")

     # task_results should correspond to task_headers.
     if len(task_results) != len(task_headers):
         logging.warning(f"Mismatch between number of task results ({len(task_results)}) and headers ({len(task_headers)}). Output alignment may be incorrect.")
         output_lines.append("\n--- Raw Task Results (Mismatch or Error) ---")
         for i, result in enumerate(task_results):
              output_lines.append(f"\n--- Result {i+1} ---")
              output_lines.append(result)
     else:
         # Process results assuming order matches headers
         for header, content in zip(task_headers, task_results):
             output_lines.append(f"\n{header.strip()}")
             content_str = str(content).strip()

             # Check if the content indicates an error from task execution
             if content_str.lower().startswith("error executing task"): # Case-insensitive check
                 output_lines.append(content_str) # Just print the error message
                 logging.warning(f"Task result for '{header.strip()}' indicates an error.")
                 continue # Move to the next result

             # Clean and format the questions from the AI output based on expected format
             cleaned_lines = [
                 line for line in content_str.split("\n")
                 # Exclude the exact header string if it's in the output
                 if header.strip() not in line and line.strip() != ""
             ]

             formatted_questions = []
             for line in cleaned_lines:
                  # Attempt to remove leading numbering (e.g., "1. ", " 2. ", "3)")
                  parts = line.split('. ', 1) # Split on ". " first
                  if len(parts) > 1 and parts[0].strip().isdigit():
                       formatted_questions.append(parts[1].strip())
                  else:
                       # If not ". ", try stripping common numbering patterns manually
                       line_stripped = line.strip()
                       if line_stripped and line_stripped[0].isdigit():
                            # Try removing leading digit followed by non-digit or punctuation
                            import re
                            match = re.match(r'^\d+\W*\s*', line_stripped)
                            if match:
                                formatted_questions.append(line_stripped[match.end():].strip())
                            else:
                                formatted_questions.append(line_stripped) # Fallback
                       else:
                           formatted_questions.append(line_stripped) # Keep if no leading digit

             if formatted_questions:
                 for idx, question in enumerate(formatted_questions, start=1):
                     output_lines.append(f"{idx}. {question}")
             else:
                 # If no questions were parsed, indicate it
                 if content_str: # If there was *any* content, but it wasn't questions
                     output_lines.append("[Task completed, but generated unexpected or unparseable output.]")
                     logging.warning(f"Task '{header.strip()}' completed but generated unexpected output:\n{content_str[:300]}...") # Log a snippet
                 else: # If content was empty after stripping
                     output_lines.append("[Task completed, generated no output content.]")
                     logging.warning(f"Task '{header.strip()}' completed but generated no output content.")


     logging.info("Output formatting complete.")
     return output_lines

# --- End Helper Functions ---


# =========================================================
# Main Application Function
# =========================================================
# Ensure this function is NOT indented, it's at the top level.
def main():
    """Main function to orchestrate the program flow."""
    logging.info("=== Meta Minds Application Started ===")

    try:
        # 1. Get file paths from user
        file_paths = get_user_input_file_paths()
        if not file_paths:
            logging.info("No file paths provided or input error. Exiting.")
            logging.info("=== Meta Minds Application Finished ===")
            return # Exit the main function

        # 2. Process the datasets (Load dataframes)
        datasets = process_datasets(file_paths)
        if not datasets:
            logging.error("No datasets could be loaded from the provided paths. Exiting.")
            logging.info("=== Meta Minds Application Finished ===")
            return # Exit if no datasets were successfully loaded

        # 3. Generate summaries for loaded datasets (Includes GPT calls for descriptions)
        # This is done BEFORE CrewAI tasks to provide context if needed later,
        # and to ensure summaries are ready for the output formatting step.
        # Store summaries in a dictionary keyed by dataset name.
        dataset_summaries = {}
        logging.info("Generating summaries for loaded datasets...")
        # Iterate through the loaded datasets to generate summaries
        for name, df in datasets:
            try:
                # generate_summary calls generate_column_descriptions which uses GPT
                summary = generate_summary(df) # generate_summary is imported from data_analyzer
                dataset_summaries[name] = summary
                logging.info(f"Summary generated for {name}")
            except Exception as e:
                logging.error(f"Error generating summary for {name}: {e}")
                # Store an error indicator in the summaries dictionary
                dataset_summaries[name] = {"error": str(e), "name": name} # Store name for easier handling in format

        logging.info("Summaries generation process finished.")
        # Note: Some summaries might have errors if GPT calls failed.

        # 4. Create agents
        schema_sleuth, question_genius = create_agents() # create_agents is imported from agents
        agents = [schema_sleuth, question_genius] # List of all agents potentially used in tasks

        # 5. Create tasks for agents based on the loaded data
        individual_tasks, individual_headers = create_tasks(datasets, schema_sleuth, question_genius) # create_tasks from tasks
        comparison_task = create_comparison_task(datasets, question_genius) # create_comparison_task from tasks

        # 6. Assemble all tasks to be run by the CrewAI process
        all_tasks = individual_tasks[:] # Start with dataset-specific tasks
        all_headers = individual_headers[:] # Start with corresponding headers

        if comparison_task:
            all_tasks.append(comparison_task)
            # Add the expected header for the comparison task
            all_headers.append("--- Comparison Questions ---") # Ensure this string matches expected_output in create_comparison_task

        if not all_tasks:
             logging.warning("No tasks were created based on the provided data. Exiting before running CrewAI.")
             # We still have summaries to output, so continue to formatting and saving
             task_results = [] # Empty list as no tasks ran
             logging.info("Skipping CrewAI execution as no tasks were created.")
        else:
            # 7. Run tasks using CrewAI
            # The run_crew_standard function handles running each task sequentially
            # and returns a list of results (strings or error messages).
            task_results = run_crew_standard(all_tasks, agents)

        # 8. Format the output
        # format_output_final uses the pre-generated summaries and task results
        # Pass dataset_summaries (which might contain errors) and results/headers from CrewAI
        formatted_output_lines = format_output_final(dataset_summaries, task_results, all_headers) # format_output_final defined above

        # 9. Save the output
        output_filename = "meta_output.txt" # Define the desired output filename
        if formatted_output_lines:
            # save_output is imported from output_handler.py
            save_output(output_filename, formatted_output_lines) # Pass filename AND lines
        else:
            logging.warning("No formatted output lines were generated to save.")


    except Exception as main_e:
        # Catch any unexpected errors that weren't handled elsewhere in the main flow
        logging.critical(f"An unexpected critical error occurred in the main workflow: {main_e}", exc_info=True)
        print(f"\nCritical Error: {main_e}")
        print("Please check the logs for more details.")

    logging.info("=== Meta Minds Application Finished ===")


# =========================================================
# Script Entry Point
# =========================================================
# This standard Python construct ensures that the main() function is called
# only when the script is executed directly (not when imported as a module).
# Ensure this block is NOT indented, it's at the top level.
if __name__ == "__main__":
    # Call the main application function
    main()