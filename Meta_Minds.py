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
        backstory="Knows how to ask questions to uncover hidden insights. Will strictly focus only on the dataset provided and avoid referencing other datasets.",
        verbose=True
    )
    return schema_sleuth, question_genius

# ==========================
# Define Tasks for Agents
# ==========================
def create_tasks(datasets, agent1, agent2):
    tasks = []
    headers = []

    for name, df in datasets:
        question_task = Task(
            description=f"""You are given a single dataset named '{name}'. ONLY use this dataset to generate questions.
Do NOT reference or compare with any other dataset, file, or external information.

Generate exactly 10 meaningful and diverse questions that a data analyst might ask.
Focus on trends, relationships, anomalies, or potential KPIs based strictly on this dataset.

Here is a sample from the dataset:

{df.head().to_string()}""",
            agent=agent2,
            expected_output=f"--- Questions for {name} ---"
        )
        tasks.append(question_task)
        headers.append(f"--- Questions for {name} ---")

    return tasks, headers

def create_comparison_task(datasets, agent):
    comparison_task = Task(
        description="""You are given multiple datasets. Generate questions that compare and contrast these datasets.
Focus on identifying trends, differences, similarities, and potential insights that can be drawn from comparing them.

Here are the datasets:
""" + "\n".join([f"Dataset '{name}':\n{df.head().to_string()}" for name, df in datasets]),
        agent=agent,
        expected_output="--- Comparison Questions ---"
    )
    return comparison_task

# ==========================
# Main Function (Entry Point)
# ==========================
def main():
    num_files = int(input("Enter number of datasets: "))
    datasets = []

    for i in range(num_files):
        file_path = input(f"Enter full path of dataset {i+1} (CSV, XLSX, or JSON): ").strip()
        df = read_file(file_path)
        datasets.append((os.path.basename(file_path), df))

    schema_sleuth, question_genius = create_agents()
    tasks, headers = create_tasks(datasets, schema_sleuth, question_genius)

    # Add comparison task
    comparison_task = create_comparison_task(datasets, question_genius)
    tasks.append(comparison_task)
    headers.append("--- Comparison Questions ---")

    logging.info("🚀 Starting Meta Minds Analysis...\n")
    results = []
    for task in tasks:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        results.append(result)

    # Save output to file
    output_lines = []

    for name, df in datasets:
        summary = generate_summary(df)
        output_lines.append(f"====== DATA SUMMARY FOR {name} ======")
        output_lines.append(f"Rows: {summary['rows']}")
        output_lines.append(f"Columns: {summary['columns']}")
        for col, info in summary["column_info"].items():
            output_lines.append(f"{col} ({info['dtype']}): {info['description']}")
        output_lines.append("")

    output_lines.append("====== GENERATED QUESTIONS ======")

    if isinstance(results, list):
        for header, content in zip(headers, results):
            if header.startswith("--- Questions for") or header.startswith("--- Comparison Questions ---"):
                output_lines.append(f"\n{header.strip()}")
                content_str = str(content).strip()
                cleaned_lines = [
                    line for line in content_str.split("\n")
                    if header.strip() not in line and line.strip() != ""
                ]
                cleaned_lines = [line.split('. ', 1)[-1] for line in cleaned_lines]
                for idx, question in enumerate(cleaned_lines, start=1):
                    output_lines.append(f"{idx}. {question.strip()}")
    else:
        output_lines.append(str(results))

    with open("meta_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\n✅ Meta Minds output saved to 'meta_output.txt'")

if __name__ == "__main__":
    main()

