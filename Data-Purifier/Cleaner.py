from crewai import Agent, Task, Crew, Process
import pandas as pd
import os
from dataprep.clean import clean_missing, clean_text
from ydata_profiling import ProfileReport
import subprocess
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
        print("Missing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def create_virtualenv(env_name):
    """Create a virtual environment."""
    subprocess.run(f"python -m venv {env_name}", shell=True)
    print(f"Virtual environment '{env_name}' created.")

def install_packages(env_name, packages):
    """Install packages in the specified virtual environment."""
    if os.name == 'nt':  # For Windows
        pip_path = os.path.join(env_name, 'Scripts', 'pip')
    else:  # For Unix or MacOS
        pip_path = os.path.join(env_name, 'bin', 'pip')
    
    for package in packages:
        subprocess.run(f"{pip_path} install {package}", shell=True)
        print(f"Installed {package} in {env_name}.")

def setup_environments():
    """Set up virtual environments for different packages."""
    # Create environments
    create_virtualenv('dataprep_env')
    create_virtualenv('instructor_env')

    # Install packages in each environment
    install_packages('dataprep_env', ['dataprep==0.4.5', 'jinja2==3.0.1', 'regex==2021.11.10', 'pandas==1.3.5', 'pydantic==1.10.2'])
    install_packages('instructor_env', ['instructor==1.7.9', 'litellm==1.60.2', 'jinja2==3.1.4', 'pydantic==1.10.2'])

    print("\nEnvironments created and packages installed. Please activate them as needed.")


# Function to verify OpenAI API key is set
def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found!")
        print("Please set your OpenAI API key using one of these methods:")
        print("\n1. Environment variable (recommended):")
        print("   Windows: set OPENAI_API_KEY=your-api-key-here")
        print("   Mac/Linux: export OPENAI_API_KEY=your-api-key-here")
        print("\n2. Direct setting in code (not recommended for security):")
        print('   os.environ["OPENAI_API_KEY"] = "your-api-key-here"')
        return False
    return True

# Set OpenAI API key (replace with your key)
os.environ["OPENAI_API_KEY"] = "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"

# Function to read different types of data files
def read_file(file_path):
    """Read different file formats (CSV, Excel, JSON) and return a pandas DataFrame"""
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Read file based on its extension
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

# Example of handling missing values with pandas
def clean_data(df):
    # Fill missing values with the mean of the column
    df.fillna(df.mean(), inplace=True)
    # Standardize text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Agent 1: Schema Sleuth
schema_sleuth = Agent(
    role="Schema Sleuth",
    goal="Analyze data and create metadata",
    backstory="I am an expert at analyzing data and extracting meaningful metadata",
    verbose=True,
    allow_delegation=False
)

# Agent 2: Cleaner
data_cleaner = Agent(
    role="Cleaner",
    goal="Clean the dataset to prepare for analysis",
    backstory="Expert in handling missing data, fixing types, removing outliers and duplicates using smart tools like DataPrep and profiling libraries.",
    verbose=True,
    allow_delegation=False
)

# Agent 3: Modifier
data_modifier = Agent(
    role="Modifier",
    goal="Modify data structure to suit analytical needs",
    backstory="Renames, recodes, and filters data to ensure clarity and usefulness",
    verbose=True,
    allow_delegation=False
)

# Agent 4: Transformer
data_transformer = Agent(
    role="Transformer",
    goal="Transform the dataset for better analytical insights",
    backstory="Expert in feature engineering, encoding, and reshaping data",
    verbose=True,
    allow_delegation=False
)

# Agent 5: Summarizer
pipeline_summarizer = Agent(
    role="Summarizer",
    goal="Summarize all tasks done, the code executed, and why each task was performed",
    backstory="Expert in documentation and process summarization",
    verbose=True,
    allow_delegation=False
)

# Define Task Templates
def get_schema_task(file_path):
    return Task(
        description=f"""
        Analyze the dataset located at '{file_path}' and create metadata including:
        - Number of rows and columns
        - Data types of all columns
        - Description of all columns
        """,
        agent=schema_sleuth,
        expected_output="A dictionary containing metadata about the data"
    )

def get_cleaning_task(file_path, questions):
    return Task(
        description=f"""
        Clean the dataset located at '{file_path}' with the goal of answering the following questions:
        {questions}

        Use DataPrep's `clean_missing()` and `clean_text()` functions to clean the data effectively.
        Also, generate a profiling report using `pandas_profiling.ProfileReport`.

        Tasks include:
        - Handling missing values (mean/median/mode as appropriate)
        - Removing duplicates
        - Fixing data types
        - Standardizing text
        - Handling outliers
        - Correcting inconsistencies

        Provide code executed, output summary (e.g., how many missing values filled), and reason for each step.
        Save the profiling report as 'profiling_report.html' in the same directory as the input file.
        """,
        agent=data_cleaner,
        expected_output="Detailed cleaning report with code, profiling summary, and justification"
    )

def get_modifying_task():
    return Task(
        description="""
        Modify the cleaned dataset by:
        - Renaming columns
        - Dropping irrelevant columns/rows
        - Filtering records
        - Recoding values
        - Fixing known typos

        Each modification should be explained with code, output effect, and justification.
        """,
        agent=data_modifier,
        expected_output="Detailed modification report with code and justification"
    )

def get_transforming_task():
    return Task(
        description="""
        Transform the modified dataset by:
        - Normalizing/standardizing numeric columns
        - Encoding categorical variables
        - Creating date-based features
        - Binning continuous variables
        - Applying log transforms if needed
        - Creating new features useful for the questions

        Provide code, output summary, and reason behind each transformation.
        """,
        agent=data_transformer,
        expected_output="Detailed transformation report with code and justification"
    )

def get_summary_task():
    return Task(
        description="""
        Summarize the outputs of all prior tasks in the following format:

        - Agent Name
        - Name of Task
        - Code Executed
        - Output Result (e.g., # rows removed, columns renamed)
        - Why the task was done this way

        Also save the final cleaned dataset as a CSV and return the file path.
        """,
        agent=pipeline_summarizer,
        expected_output="Structured summary report and file path of cleaned dataset"
    )

# Build the Crew
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

# Main Entry Point
def load_questions():
    print("Enter the questions one by one. Type 'done' when finished:")
    questions = []
    while True:
        q = input("- ").strip()
        if q.lower() == 'done':
            break
        questions.append(q)
    return questions

if __name__ == "__main__":
    print("\n=== Setting Up Virtual Environments ===")
    setup_environments()    
    
    print("\n=== Data Processing Crew Runner ===")
    num_datasets = int(input("Enter the number of datasets to process: ").strip())

    for i in range(num_datasets):
        print(f"\nDataset {i + 1}:")
        file_path = input("Enter path to your dataset file: ").strip()

        if not os.path.exists(file_path):
            print("\n[Error] File not found. Please check the path and try again.")
            continue

        questions = load_questions()

        print("\nRunning the crew...\n")
        crew = build_crew(file_path, questions)
        results = crew.kickoff()

        print("\n=== Crew Execution Complete ===")
        print(results)



