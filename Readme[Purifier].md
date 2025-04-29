
# 🧹 Cleaner Agent (CrewAI Data Cleaning Pipeline)

This project is an intelligent data cleaning, modification, and transformation pipeline powered by **CrewAI** agents. It automatically processes raw datasets using expert-level AI agents and structured logic, based on metadata and explainable steps.


## 🚀 Key Features

### ✅ Automated Data Cleaning
- Removes duplicates
- Replaces empty strings
- Converts columns to datetime and numeric types
- Handles missing values intelligently
- Standardizes date formats
- Detects and removes outliers using Isolation Forest

### 🛠️ Data Modification
- Renames columns to snake_case
- Reorders columns (e.g., user_id first)
- Drops irrelevant or temporary columns
- Filters data (e.g., active users)
- Aggregates metrics (e.g., average steps per user)
- Splits combined columns (e.g., full_name to first_name and last_name)

### 🔁 Data Transformation
- Normalizes numeric values
- Encodes categorical columns (LabelEncoder)
- Applies log transformations
- Creates derived features (e.g., BMI)
- Performs binning (e.g., age group buckets)

### 🧠 Metadata-Aware Processing
- Reads `meta_output.txt` to inform cleaning logic
- Converts columns dynamically based on metadata keywords like `timestamp`

### 📒 Summary and Feedback
- Saves all processing steps and user feedback into `process_report.md`
- Optional user feedback is collected at the end of the process

### 🤖 Powered by CrewAI Agents
- `Cleaner Agent`: Cleans raw data
- `Modifier Agent`: Modifies columns and rows
- `Transformer Agent`: Applies transformations
- `Summarizer Agent`: Summarizes the pipeline actions


## 🧩 Workflow

```
flowchart LR
    A[User Provides CSV + meta_output.txt] --> B[Cleaner Agent]
    B --> C[Modifier Agent]
    C --> D[Transformer Agent]
    D --> E[Summarizer Agent]
    E --> F[Final CSV + Markdown Report + HTML Profiling]
```


## 🛠️ Setup Instructions

### 1. Clone the Repo & Install Dependencies
```bash
pip install pandas numpy scikit-learn pandas-profiling crewai openai
```

### 2. Add Your OpenAI API Key
Set your OpenAI key in the script:
```
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 3. Prepare Your Files
- Dataset(s) in CSV format
- A `meta_output.txt` file with metadata text (used by the agent)

### 4. Run the Script
```
python CleanerAgent.py
```

Follow the on-screen prompts to:
- Enter number of datasets
- Provide each dataset path


## 📁 Outputs

| Output File | Description |
|-------------|-------------|
| `final_dataset_<i>.csv` | Cleaned, modified, transformed dataset |
| `process_report.md`     | Step-by-step process log and user feedback |
| `profiling_report.html` | Full profiling report of the original data |
| `customized_profiling_report.html` | Customized profiling after transformation |

## 📌 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- pandas_profiling
- crewai
- openai

## 🧠 Future Improvements

- Add data validation checks before processing
- Build GUI interface using Streamlit or Gradio
- Integrate with database connectors (PostgreSQL, Snowflake, etc.)
- Enable auto-target detection for feature selection

