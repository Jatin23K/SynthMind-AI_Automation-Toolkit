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
    descriptions = {}
    for column in data.columns:
        sample_values = data[column].dropna().astype(str).head(3).tolist()
        sample_text = ", ".join(sample_values)
        prompt = (
            f"Column name: {column}\n"
            f"Sample values: {sample_text}\n"
            f"Based on the column name and sample values, write a short, clear description of this column:"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
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
    meta = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": {}
    }
    descriptions = generate_column_descriptions(df)
    for col in df.columns:
        meta["column_info"][col] = {
            "dtype": str(df[col].dtype),
            "description": descriptions[col]
        }
    return meta


# ==========================
# Define CrewAI Agents
# ==========================
def create_agents():
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
    file_path = input("Enter the full path of your dataset (CSV, XLSX, or JSON): ").strip()
    df = read_file(file_path)
    schema_sleuth, question_genius = create_agents()
    tasks = create_tasks(df, os.path.basename(file_path), schema_sleuth, question_genius)

    crew = Crew(
        agents=[schema_sleuth, question_genius],
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )

    logging.info("🚀 Starting Meta Minds Analysis...\n")
    result = crew.kickoff()
    summary = generate_summary(df)

    # Save output to a text file
    output_lines = ["====== DATA SUMMARY ======", f"Rows: {summary['rows']}", f"Columns: {summary['columns']}", ""]
    for col, info in summary["column_info"].items():
        output_lines.append(f"{col} ({info['dtype']}): {info['description']}")
    output_lines.append("\n====== GENERATED QUESTIONS ======")
    if isinstance(result, list):
        for q in result:
            output_lines.append(f"- {q}")
    else:
        output_lines.append(str(result))

    with open("meta_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\n✅ Meta Minds output saved to 'meta_output.txt'")

if __name__ == "__main__":
    main()
