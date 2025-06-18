from crewai import Task
import pandas as pd # Needed for DataFrame type hinting
import logging # Although not strictly needed inside tasks, useful for logging task creation

def create_tasks(datasets: list[tuple[str, pd.DataFrame]], agent1, agent2) -> tuple[list[Task], list[str]]:
    """Creates individual tasks for the AI agents to generate questions for each dataset.

    Args:
        datasets (list): A list of tuples containing the dataset name (str) and DataFrame (pd.DataFrame).
        agent1 (crewai.Agent): The schema sleuth agent (though not used in the *current* task definition,
                                included for completeness if schema tasks were added).
        agent2 (crewai.Agent): The question genius agent.

    Returns:
        tuple: A tuple containing the list of generated crewai.Task objects
               and the list of corresponding header strings for output.
    """
    logging.info(f"Creating individual dataset analysis tasks for {len(datasets)} dataset(s)...")
    tasks = []
    headers = []

    # Note: The original code had both agents as arguments but only used agent2 (question_genius)
    # for the question generation task. The schema analysis/description was done outside CrewAI.
    # If you wanted a CrewAI task for schema analysis, you would define it here as well,
    # potentially using agent1 (schema_sleuth). For now, adhering to the original task structure.

    for name, df in datasets:
        # Limit sample to avoid exceeding token limits for very wide dataframes
        sample_string = df.head().to_string()
        if len(sample_string) > 2000: # Arbitrary limit, adjust as needed
             sample_string = df.head().to_string()[:2000] + "\n[...truncated...]"
             logging.warning(f"Sample for task description for '{name}' was truncated.")


        question_task = Task(
            description=f"""You are given a single dataset named '{name}'. Your goal is to generate insightful questions for data analysis.
            
**Constraints:**
1. **STRICTLY use ONLY the data provided in this dataset ('{name}').**
2. Do NOT reference or compare with any other dataset, file, or external information.
3. Generate exactly 20 distinct, meaningful, and diverse analytical questions.
4. Questions should focus on identifying trends, relationships, anomalies, potential KPIs, or areas for deeper investigation *within* this dataset.
5. Ensure questions are clear and actionable for a data analyst.

Here is a sample from the dataset to help you understand its content:

{sample_string}""",
            agent=agent2, # Assign this task to the question_genius agent
            expected_output=f"""A numbered list (1. 2. etc.) of exactly 20 analytical questions based *only* on the '{name}' dataset.
            Start your output with the exact string: "--- Questions for {name} ---" """,
            human_input=False # Typically tasks don't need human input in this flow
        )
        tasks.append(question_task)
        headers.append(f"--- Questions for {name} ---")

    logging.info(f"Created {len(tasks)} individual dataset analysis tasks.")
    return tasks, headers


def create_comparison_task(datasets: list[tuple[str, pd.DataFrame]], agent) -> Task | None:
    """Creates a task for the AI agent to generate comparison questions across multiple datasets.

    This task is only created if there is more than one dataset provided.

    Args:
        datasets (list): A list of tuples containing the dataset name (str) and DataFrame (pd.DataFrame).
        agent (crewai.Agent): The question genius agent.

    Returns:
        crewai.Task | None: The comparison task if more than one dataset exists, otherwise None.
    """
    if len(datasets) <= 1:
        logging.info("Only one dataset provided, skipping comparison task creation.")
        return None # No comparison needed for a single file

    logging.info(f"Creating comparison analysis task for {len(datasets)} datasets...")

    # Concatenate samples from all datasets for the comparison prompt
    comparison_sample_string = ""
    for name, df in datasets:
         sample_string = df.head().to_string()
         if len(sample_string) > 1000: # Truncate samples for comparison prompt
             sample_string = df.head().to_string()[:1000] + "\n[...truncated...]"
             logging.warning(f"Sample for comparison task description for '{name}' was truncated.")
         comparison_sample_string += f"\nDataset '{name}':\n{sample_string}\n"
         comparison_sample_string += "-"*20 + "\n" # Separator

    comparison_task = Task(
        description=f"""You are given multiple datasets with the goal of generating questions that compare and contrast them.

**Constraints:**
1. Generate exactly 15 meaningful and diverse analytical questions.
2. Questions MUST focus on identifying trends, differences, similarities, or potential insights that can be drawn *specifically* by comparing and contrasting the provided datasets.
3. Do NOT generate questions that are specific to only one dataset; focus on comparative analysis.
4. Do NOT reference any external data or knowledge.
5. Ensure questions are clear and actionable for a data analyst performing a comparative study.

Here are samples from the datasets:

{comparison_sample_string}""",
        agent=agent, # Assign this task to the question_genius agent
        expected_output="""A numbered list (1. 2. etc.) of 15 analytical questions that compare and contrast the provided datasets.
        Start your output with the exact string: "--- Comparison Questions ---" """,
        human_input=False # Typically tasks don't need human input in this flow
    )
    logging.info("Comparison analysis task created.")
    return comparison_task