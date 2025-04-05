## Once dataset is upload agent understood the data and generate metadata (number of rows, columns, column names, data types), insightful questions


# Import required libraries
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from typing import Dict, List

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
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
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

# Function to generate descriptions for each column in the dataset
def generate_column_descriptions(data):
    """Generate meaningful descriptions for each column based on the data"""
    # Predefined descriptions for common columns
    descriptions = {
        "date": "The date the data was recorded, formatted as MM/DD/YYYY.",
        "steps": "The total number of steps taken on the recorded date.",
        "distance_km": "The distance covered in kilometers on the recorded date.",
        "calories_burned": "The total calories burned on the recorded date.",
        "active_minutes": "The number of minutes actively engaged in physical activities on the recorded date.",
        "sleep_hours": "The total hours of sleep recorded on the specific date.",
        "water_intake_liters": "The amount of water consumed in liters on the recorded date.",
        "mood": "The emotional state recorded for the individual on that day, categorized as 'stressed', 'sad', 'tired', etc."
    }
    
    # Create descriptions for actual columns in the data
    actual_descriptions = {}
    for column in data.columns:
        if column in descriptions:
            actual_descriptions[column] = descriptions[column]
        else:
            # Generate generic description for unknown columns
            actual_descriptions[column] = f"Data recorded for {column.replace('_', ' ').lower()}."
    
    return actual_descriptions

# Function to generate summary statistics and metadata about the dataset
def generate_data_summary(data):
    """Generate a summary of the data including rows, columns, data types, and descriptions"""
    summary = {
        "rows": len(data),
        "columns": len(data.columns),
        "column_info": {}
    }
    
    # Get column descriptions
    descriptions = generate_column_descriptions(data)
    
    # Generate detailed information for each column
    for column in data.columns:
        summary["column_info"][column] = {
            "dtype": str(data[column].dtype),
            "description": descriptions.get(column, f"Column containing {data[column].dtype} data")
        }
    
    return summary

# Create AI agents with specific roles and capabilities
def create_agents():
    # Agent 1: Schema Sleuth - Analyzes data structure and metadata
    data_analyzer = Agent(
        role="Schema Sleuth",
        goal="Analyze data and create metadata",
        backstory="I am an expert at analyzing data and extracting meaningful metadata",
        verbose=True,
        allow_delegation=False
    )

    # Agent 2: Curious Catalyst - Generates analytical questions
    question_generator = Agent(
        role="Curious Catalyst",
        goal="Generate analytical questions based on data analysis",
        backstory="I am an expert at formulating insightful analytical questions",
        verbose=True,
        allow_delegation=False
    )

    # Agent 3: Data Comparator - Compares multiple datasets
    data_comparator = Agent(
        role="Data Comparator",
        goal="Compare multiple datasets and identify key differences and relationships",
        backstory="I am an expert at comparing datasets and finding meaningful patterns and differences",
        verbose=True,
        allow_delegation=False
    )

    return data_analyzer, question_generator, data_comparator

# Create tasks for analyzing a single dataset
def create_analysis_tasks(data_analyzer, question_generator, data, dataset_name=""):
    # Task 1: Analyze the dataset structure and content
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

    # Task 2: Generate analytical questions about the data
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

# Create task for comparing multiple datasets
def create_multi_comparison_task(data_comparator, datasets_dict):
    """Create a task to compare multiple datasets"""
    # Generate a string containing information about all datasets
    datasets_info = "\n\n".join([
        f"Dataset {name}:\n{data.head().to_string()}\nShape: {data.shape}"
        for name, data in datasets_dict.items()
    ])
    
    # Create the comparison task
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

# Function to analyze a single dataset
def analyze_single_dataset(data, dataset_name=""):
    """Analyze a single dataset using AI agents"""
    # Create agents and tasks
    data_analyzer, question_generator, _ = create_agents()
    tasks = create_analysis_tasks(data_analyzer, question_generator, data, dataset_name)
    
    # Create and run the crew
    crew = Crew(
        agents=[data_analyzer, question_generator],
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )
    return crew.kickoff()

# Function to analyze multiple datasets
def analyze_multiple_datasets(datasets_dict):
    """Analyze multiple datasets and provide comparisons"""
    # Create all agents
    data_analyzer, question_generator, data_comparator = create_agents()
    
    # Store results for each dataset
    analysis_results = {}
    data_summaries = {}
    
    # Analyze each dataset individually
    for name, data in datasets_dict.items():
        print(f"\nAnalyzing {name}...")
        analysis_results[name] = analyze_single_dataset(data, name)
        data_summaries[name] = generate_data_summary(data)
    
    # Compare all datasets if there are more than one
    if len(datasets_dict) > 1:
        print("\nComparing all datasets...")
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

# Function to get file paths from user for multiple datasets
def get_multiple_file_paths(num_datasets):
    """Get file paths for multiple datasets from user"""
    datasets_dict = {}
    
    # Get file path for each dataset
    for i in range(num_datasets):
        # Get file path from user
        print(f"\nDataset {i+1}:")
        file_path = input("Enter the complete file path: ").strip()
        
        try:
            # Try to read the file
            print(f"\nReading file: {file_path}")
            data = read_file(file_path)
            print(f"File read successfully. Found {len(data)} rows and {len(data.columns)} columns.")
            datasets_dict[f"Dataset {i+1}"] = data
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    return datasets_dict

# Main execution block
if __name__ == "__main__":
    # Check dependencies and API key
    if not check_dependencies() or not check_api_key():
        exit(1)
    
    # Ask user for number of datasets to analyze
    print("\nHow many datasets would you like to analyze?")
    try:
        num_datasets = int(input("Enter a number (1 or more): ").strip())
        if num_datasets < 1:
            raise ValueError("Number of datasets must be 1 or more")
    except ValueError as e:
        print(f"Invalid input: {str(e)}")
        exit(1)
    
    try:
        # Get datasets from user
        datasets_dict = get_multiple_file_paths(num_datasets)
        if not datasets_dict:
            print("Error reading one or more files. Please check the file paths and try again.")
            exit(1)
        
        # Analyze datasets and get results
        analysis_results, data_summaries, comparison_result = analyze_multiple_datasets(datasets_dict)
        
        # Print results for each dataset
        for name in datasets_dict.keys():
            print(f"\n{'='*50}")
            print(f"{name.upper()} ANALYSIS")
            print('='*50)
            
            summary = data_summaries[name]
            print(f"Number of rows: {summary['rows']}")
            print(f"Number of columns: {summary['columns']}")
            
            print("\nColumn Descriptions:")
            descriptions = generate_column_descriptions(datasets_dict[name])
            for column, description in descriptions.items():
                print(f"\n{column}:")
                print(f"Type: {summary['column_info'][column]['dtype']}")
                print(f"Description: {description}")
            
            print("\nGenerated Questions:")
            print(analysis_results[name])
        
        # Print comparison results if there are multiple datasets
        if num_datasets > 1 and comparison_result:
            print(f"\n{'='*50}")
            print("DATASETS COMPARISON")
            print('='*50)
            print(comparison_result)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("If this is a permission error, make sure you have access to the files.")
        print("If this is a format error, make sure all files are valid CSV, Excel, or JSON files.")
