# ==========================
# Meta Minds: AI-Powered Dataset Analyzer
# ==========================
# This code loads your dataset, understands its structure,
# describes each column using GPT (OpenAI), and generates 15 smart questions all automatically using CrewAI.

import os  # For handling file paths and environment variables
import pandas as pd  # To read and manage data tables (like Excel/CSV)
from crewai import Agent, Task, Crew, Process  # CrewAI: to define AI agents and workflows
import logging  # Helps track what the code is doing and catch errors
from openai import OpenAI  # The latest OpenAI library for chatting with GPT

# ==========================
# Setup Logging
# ==========================
# This makes sure that OpenAI's internal logs don't flood our screen
logging.getLogger("openai").setLevel(logging.WARNING)
# Setup how messages will appear in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================
# API Key Setup
# ==========================
# NOTE:
# If your editor shows a red underline here, don't worry!
# It's just warning you that the OpenAI API key is missing or fake.
# Replace the placeholder below with your actual OpenAI API key or set it in your environment.

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxx"  # Replace this with your actual key


# Create the GPT client (v1 style)
client = OpenAI()

# ==========================
# Generate Descriptions Using GPT
# ==========================
def generate_column_descriptions(data):
    """This function sends sample values from each column to GPT
    so it can describe what the column likely represents."""
    descriptions = {}

    for column in data.columns:
        sample_values = data[column].dropna().astype(str).head(3).tolist()
        sample_text = ", ".join(sample_values)

        # Create a custom instruction for GPT
        prompt = (
            f"Column name: {column}\n"
            f"Sample values: {sample_text}\n"
            f"Based on the column name and sample values, write a short, clear description of this column:"
        )

        try:
            # Ask GPT to describe this column
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            descriptions[column] = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Failed to generate description for '{column}': {e}")
            descriptions[column] = f"Sample values: {sample_text}"

    return descriptions

# ==========================
# Load the Dataset
# ==========================
def read_file(file_path):
    """This reads your file and turns it into a data table Python can work with"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".xlsx":
        return pd.read_excel(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ==========================
# Create Summary (Rows, Columns, Descriptions)
# ==========================
def generate_summary(df):
    """Creates a summary of your dataset: row/column count + column descriptions"""
    meta = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": {}
    }
    descriptions = generate_column_descriptions(df)
    for col in df.columns:
        meta["column_info"][col] = {
            "dtype": str(df[col].dtype),  # The type of data in the column (e.g., numbers, text)
            "description": descriptions[col]  # Description from GPT
        }
    return meta

# ==========================
# Define CrewAI Agents
# ==========================
def create_agents():
    """These are your AI team members with specific roles"""
    schema_sleuth = Agent(
        role="Schema Sleuth",
        goal="Analyze data and describe schema",
        backstory="Expert in identifying data structure and metadata",
        verbose=True
    )
    question_genius = Agent(
        role="Curious Catalyst",
        goal="Generate insightful questions from data",
        backstory="Knows how to ask questions to uncover hidden insights",
        verbose=True
    )
    return schema_sleuth, question_genius

# ==========================
# Define Tasks for Agents
# ==========================
def create_tasks(data, name, agent1, agent2):
    """Each agent gets a task with specific instructions"""
    analyze_task = Task(
        description=f"""Analyze the dataset '{name}' and produce metadata including row/column count,
        column data types, and purpose of each column.
        Sample:
        {data.head().to_string()}""",
        agent=agent1,
        expected_output="Structured metadata with column names, types, and descriptions"
    )

    question_task = Task(
        description=f"""Based on the data structure of '{name}', generate exactly 15 insightful and varied questions
        that a data analyst might ask to explore patterns, trends, or anomalies in the dataset.
        The questions should reflect real-world data reasoning and be relevant to the column structure and values.""",
        agent=agent2,
        expected_output="A numbered list of 15 exploratory data analysis questions"
    )

    return [analyze_task, question_task]

# ==========================
# Main Function (Entry Point)
# ==========================
def main():
    """This is what runs when you launch the program"""
    file_path = input("Enter the full path of your dataset (CSV, XLSX, or JSON): ").strip()
    df = read_file(file_path)
    schema_sleuth, question_genius = create_agents()
    tasks = create_tasks(df, os.path.basename(file_path), schema_sleuth, question_genius)

    # Create the AI crew with our agents and tasks
    crew = Crew(
        agents=[schema_sleuth, question_genius],
        tasks=tasks,
        verbose=True,
        process=Process.sequential  # Run one after the other
    )

    logging.info("🚀 Starting Meta Minds Analysis...\n")
    result = crew.kickoff()  # Begin the task execution
    summary = generate_summary(df)  # Create a summary from the dataset

    # Print results to the screen
    print("\n====== DATA SUMMARY ======")
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")
    for col, info in summary["column_info"].items():
        print(f"\n{col} ({info['dtype']}):\n  {info['description']}")

    print("\n====== GENERATED QUESTIONS ======")
    print(result)

# ==========================
# Run the Program
# ==========================
if __name__ == "__main__":
    main()
