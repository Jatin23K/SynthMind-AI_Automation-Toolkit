# Import required libraries
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from typing import Dict, List
import logging
from openai import OpenAI

# Set up logging instead of print for better control/logging in production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if all required packages are installed
def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'crewai': 'crewai',
        'pandas': 'pandas',
        'langchain': 'langchain'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.error("Missing required packages. Please install them using:")
        logging.error(f"pip install {' '.join(missing_packages)}")
        return False
    return True

# Function to verify OpenAI API key is set
def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OpenAI API key not found!")
        logging.warning("Please set your OpenAI API key using one of these methods:")
        logging.warning("1. Environment variable (recommended):")
        logging.warning("   Windows: set OPENAI_API_KEY=your-api-key-here")
        logging.warning("   Mac/Linux: export OPENAI_API_KEY=your-api-key-here")
        logging.warning("2. Direct setting in code (not recommended for security):")
        logging.warning('   os.environ["OPENAI_API_KEY"] = "your-api-key-here"')
        return False
    return True

# Securely set OpenAI API key only if not already set
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Function to read different types of data files
def read_file(file_path):
    """Read different file formats (CSV, Excel, JSON) and return a pandas DataFrame"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    elif file_extension == '.pdf':
        raise NotImplementedError("PDF support not implemented")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# Generate dynamic descriptions for each column using GPT
import openai

def generate_column_descriptions(data):
    """Use GPT to generate meaningful descriptions for each column"""
    descriptions = {}
    for column in data.columns:
        # Prepare sample values from the column
        sample_values = data[column].dropna().astype(str).head(3).tolist()
        sample_text = ", ".join(sample_values)

        # Prompt for GPT to generate a description
        prompt = (
            f"Column name: {column}\n"
            f"Sample values: {sample_text}\n"
            f"Based on the column name and sample values, write a short, clear description of this column:"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            description = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Failed to generate description for column '{column}': {e}")
            description = f"Column '{column}' with sample values: {sample_text}"

        descriptions[column] = description
    return descriptions

# Generate dataset metadata like row/column count, datatypes, and column insights
def generate_data_summary(data):
    summary = {
        "rows": len(data),
        "columns": len(data.columns),
        "column_info": {}
    }

    descriptions = generate_column_descriptions(data)

    for column in data.columns:
        summary["column_info"][column] = {
            "dtype": str(data[column].dtype),
            "description": descriptions.get(column, f"Column containing {data[column].dtype} data")
        }

    return summary

# Create AI agents with fixed roles to analyze, question, and compare datasets
def create_agents():
    data_analyzer = Agent(
        role="Schema Sleuth",
        goal="Analyze data and create metadata",
        backstory="I am an expert at analyzing data and extracting meaningful metadata",
        verbose=True,
        allow_delegation=False
    )

    question_generator = Agent(
        role="Curious Catalyst",
        goal="Generate analytical questions based on data analysis",
        backstory="I am an expert at formulating insightful analytical questions",
        verbose=True,
        allow_delegation=False
    )

    data_comparator = Agent(
        role="Data Comparator",
        goal="Compare multiple datasets and identify key differences and relationships",
        backstory="I am an expert at comparing datasets and finding meaningful patterns and differences",
        verbose=True,
        allow_delegation=False
    )

    return data_analyzer, question_generator, data_comparator

# Build dataset analysis and question-generation tasks
def create_analysis_tasks(data_analyzer, question_generator, data, dataset_name=""):
    analyze_task = Task(
        description=f"""
        Analyze the following data{f' for {dataset_name}' if dataset_name else ''} and create metadata including:
        - Number of rows and columns
        - Data types of all columns
        - Description of all columns

        Data:
        {data.head().to_string()}

        Data Info:
        Rows: {len(data)}
        Columns: {', '.join(data.columns)}
        """,
        agent=data_analyzer,
        expected_output="A dictionary containing metadata about the data"
    )

    question_task = Task(
        description=f"""
        Based on the data analysis{f' for {dataset_name}' if dataset_name else ''}, generate analytical and logical questions
        from the perspective of a data analyst. These questions should help
        understand patterns, trends, and insights in the data.
        """,
        agent=question_generator,
        expected_output="A list of analytical questions about the data"
    )

    return [analyze_task, question_task]

# Create comparison task for multiple datasets using Data Comparator agent
def create_multi_comparison_task(data_comparator, datasets_dict):
    datasets_info = "

".join([
        f"Dataset {name}:
{data.head().to_string()}
Shape: {data.shape}"
        for name, data in datasets_dict.items()
    ])

    return Task(
        description=f"""
        Compare the following datasets and identify key differences, relationships, and patterns:

        {datasets_info}

        Consider:
        - Common and unique columns across all datasets
        - Data distributions and patterns
        - Potential relationships between the datasets
        - Key differences in patterns or trends
        - Overall insights from comparing all datasets together
        """,
        agent=data_comparator,
        expected_output="A comprehensive comparison analysis of all datasets"
    )

# Analyze a single dataset by kicking off Schema Sleuth and Curious Catalyst
def analyze_single_dataset(data, dataset_name=""):
    data_analyzer, question_generator, _ = create_agents()
    tasks = create_analysis_tasks(data_analyzer, question_generator, data, dataset_name)

    crew = Crew(
        agents=[data_analyzer, question_generator],
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )
    return crew.kickoff()

# Analyze multiple datasets and trigger comparison if >1 dataset
def analyze_multiple_datasets(datasets_dict):
    data_analyzer, question_generator, data_comparator = create_agents()

    analysis_results = {}
    data_summaries = {}

    for name, data in datasets_dict.items():
        logging.info(f"Analyzing {name}...")
        analysis_results[name] = analyze_single_dataset(data, name)
        data_summaries[name] = generate_data_summary(data)

    if len(datasets_dict) > 1:
        logging.info("Comparing all datasets...")
        comparison_task = create_multi_comparison_task(data_comparator, datasets_dict)
        comparison_crew = Crew(
            agents=[data_comparator],
            tasks=[comparison_task],
            verbose=True,
            process=Process.sequential
        )
        comparison_result = comparison_crew.kickoff()
    else:
        comparison_result = None

    return analysis_results, data_summaries, comparison_result

# Ask user to provide multiple dataset paths and validate file reads
def get_multiple_file_paths(num_datasets):
    datasets_dict = {}

    for i in range(num_datasets):
        logging.info(f"
Dataset {i+1}:")
        file_path = input("Enter the complete file path: ").strip()

        try:
            logging.info(f"Reading file: {file_path}")
            data = read_file(file_path)
            logging.info(f"File read successfully. Found {len(data)} rows and {len(data.columns)} columns.")
            datasets_dict[f"Dataset {i+1}"] = data
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            return None

    return datasets_dict

# Main entry point of program
if __name__ == "__main__":
    if not check_dependencies() or not check_api_key():
        exit(1)

    logging.info("
How many datasets would you like to analyze?")
    try:
        num_datasets = int(input("Enter a number (1 or more): ").strip())
        if num_datasets < 1:
            raise ValueError("Number of datasets must be 1 or more")
    except ValueError as e:
        logging.error(f"Invalid input: {str(e)}")
        exit(1)

    try:
        datasets_dict = get_multiple_file_paths(num_datasets)
        if not datasets_dict:
            logging.error("Error reading one or more files. Please check the file paths and try again.")
            exit(1)

        analysis_results, data_summaries, comparison_result = analyze_multiple_datasets(datasets_dict)

        for name in datasets_dict.keys():
            logging.info(f"
{'='*50}
{name.upper()} ANALYSIS
{'='*50}")
            summary = data_summaries[name]
            logging.info(f"Number of rows: {summary['rows']}")
            logging.info(f"Number of columns: {summary['columns']}")

            logging.info("
Column Descriptions:")
            descriptions = generate_column_descriptions(datasets_dict[name])
            for column, description in descriptions.items():
                logging.info(f"
{column}:")
                logging.info(f"Type: {summary['column_info'][column]['dtype']}")
                logging.info(f"Description: {description}")

            logging.info("
Generated Questions:")
            logging.info(analysis_results[name])

        if num_datasets > 1 and comparison_result:
            logging.info(f"
{'='*50}
DATASETS COMPARISON
{'='*50}")
            logging.info(comparison_result)

    except Exception as e:
        logging.error(f"
Error: {str(e)}")
        logging.error("If this is a permission error, make sure you have access to the files.")
        logging.error("If this is a format error, make sure all files are valid CSV, Excel, or JSON files.")
