from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# === OpenAI Thinking Tool ===
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key="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"
)

llm_tool = Tool(
    name="think_with_openai",
    description="Use OpenAI LLM to reflect, decide, or explain tasks",
    func=lambda prompt: llm.predict(prompt)
)

def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# === Agents with Thinking Power ===
schema_sleuth = Agent(
    role="Schema Sleuth",
    goal="Analyze data and create metadata",
    backstory="I extract meaningful insights from the structure of the dataset.",
    verbose=True,
    tools=[llm_tool]
)

data_cleaner = Agent(
    role="Cleaner",
    goal="Clean the dataset smartly using analyst thinking",
    backstory="Handles missing data, outliers, and standardization using reasoning and tools.",
    verbose=True,
    tools=[llm_tool]
)

data_modifier = Agent(
    role="Modifier",
    goal="Modify dataset structure to match analytical needs",
    backstory="Refines the structure: filters, renames, and improves clarity.",
    verbose=True,
    tools=[llm_tool]
)

data_transformer = Agent(
    role="Transformer",
    goal="Transform data for insights and modeling",
    backstory="Specialist in encoding, feature engineering, and scaling",
    verbose=True,
    tools=[llm_tool]
)

pipeline_summarizer = Agent(
    role="Summarizer",
    goal="Summarize all steps and code",
    backstory="Logs every step, code used, and decisions taken.",
    verbose=True,
    tools=[llm_tool]
)

# === Tasks ===
def get_schema_task(file_path):
    df = read_file(file_path)
    preview = df.head(10).to_string()
    return Task(
        description=f"""
        Analyze the structure of the dataset at '{file_path}'.
        Generate metadata:
        - Number of rows and columns
        - Data types
        - Descriptions of each column

        Dataset preview:
        {preview}
        """,
        agent=schema_sleuth,
        expected_output="Dictionary with metadata"
    )

def get_cleaning_task(file_path, questions):
    df = read_file(file_path)
    preview = df.head(10).to_string()
    return Task(
        description=f"""
        Clean the dataset at '{file_path}'. Use your judgment as a data analyst.

        Questions to guide cleaning:
        {questions}

        Consider:
        - Missing values (SimpleImputer with mean/median for numeric, most frequent for categorical)
        - Duplicates
        - Outliers (z-score method or IQR-based)
        - Data types
        - Text cleanup (strip and lower)

        Justify every choice and show the before/after impact.

        Dataset preview:
        {preview}
        """,
        agent=data_cleaner,
        expected_output="Cleaned data logic, code used, and reasoning"
    )

def get_modifying_task():
    return Task(
        description="""
        Modify the dataset to improve clarity and usefulness.
        Tasks you may consider:
        - Rename confusing columns
        - Drop irrelevant rows/columns
        - Recode values
        - Filter data as needed

        Provide code, before/after explanation, and justification.
        """,
        agent=data_modifier,
        expected_output="Modification report with code and justifications"
    )

def get_transforming_task():
    return Task(
        description="""
        Apply transformations for analytics or modeling:
        - Normalize or standardize numeric columns
        - Encode categorical features (OneHot or Label)
        - Create new features
        - Log transform if necessary
        - Binning or date features

        Use your reasoning to choose what's helpful.
        """,
        agent=data_transformer,
        expected_output="Transformation logic, code, and reasons"
    )

def get_summary_task():
    return Task(
        description="""
        Summarize all completed tasks.
        For each:
        - Agent name
        - Task name
        - Code executed
        - Output result (e.g., columns dropped)
        - Justification

        Save final cleaned DataFrame as CSV and return its file path.
        """,
        agent=pipeline_summarizer,
        expected_output="Process summary and saved dataset path"
    )

# === Crew Builder ===
def build_crew(file_path, questions):
    tasks = [
        get_schema_task(file_path),
        get_cleaning_task(file_path, questions),
        get_modifying_task(),
        get_transforming_task(),
        get_summary_task()
    ]
    return Crew(
        agents=[schema_sleuth, data_cleaner, data_modifier, data_transformer, pipeline_summarizer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

# === CLI Runner ===
def load_questions():
    print("Enter questions to guide the analysis (type 'done' to finish):")
    questions = []
    while True:
        q = input("- ").strip()
        if q.lower() == 'done':
            break
        questions.append(q)
    return questions

import openai

# === Ensure OpenAI API Key is Set ===
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ").strip()

if __name__ == "__main__":
    print("\n=== Data Purifier AI Crew ===")
    file_path = input("Enter path to dataset: ").strip()

    if not os.path.exists(file_path):
        print("[Error] File not found. Please check the path.")
        exit()

    questions = load_questions()
    crew = build_crew(file_path, questions)
    result = crew.kickoff()

    print("\n=== Process Completed ===")
    print(result)
