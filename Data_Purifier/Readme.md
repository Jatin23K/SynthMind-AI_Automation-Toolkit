# Data Purification System

![Data Purification System Architecture](images/architecture.png) <!-- Placeholder for an architecture diagram if available -->

A sophisticated, AI-powered data purification system designed to replicate the meticulous behavior of a senior data analyst. This system provides automated data cleaning, transformation, and analysis capabilities with advanced features, comprehensive reporting, and adaptive processing.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Internal Data Handling (Feather Files)](#internal-data-handling-feather-files)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This system automates the complex process of data purification, ensuring data quality, consistency, and readiness for advanced analytical tasks and machine learning models. It employs a multi-agent architecture where specialized AI agents collaborate, validate each other's work, and adapt their strategies based on real-time feedback.

## Key Features

- **Advanced Multi-Agent Architecture**: Features highly specialized and intelligent agents (Meta-Analyzer, Cleaner, Modifier, Transformer, Validators, Orchestrator, Process Recorder) with adaptive capabilities and enhanced reasoning.
-   **Adaptive Data Processing**: Agents dynamically select optimal methods for cleaning (e.g., adaptive imputation, intelligent outlier handling, robust inconsistency resolution), modification (e.g., adaptive feature engineering, scaling), and transformation (e.g., adaptive categorical encoding) based on data characteristics and LLM-driven insights.
-   **Dynamic Pipeline Adjustment**: The Orchestrator agent dynamically adapts the data processing pipeline based on real-time feedback from validation agents, allowing for iterative refinement and self-correction.
-   **Enhanced Error Handling and Recovery**: Robust error handling with specific exception catches, detailed logging, and retry mechanisms for transient issues, ensuring pipeline resilience.
-   **Intelligent Feedback Loops & Learning**: Agents learn from validation outcomes, enabling adaptive strategies for subsequent operations within the same pipeline run, improving overall data quality over time.
-   **Comprehensive and Human-Readable Reporting**: The `ProcessRecorderAgent` generates highly detailed JSON reports, explaining *what* operations were performed, *why* (including adaptive choices), and their *impact* on data quality, making the entire process transparent and understandable.
-   **Modular Design**: Easy to extend with new agents, data processing techniques, and utilities, promoting maintainability and scalability.
-   **Comprehensive Testing**: Unit tests for all major components ensure reliability and correctness of operations.
-   **Configuration Management**: Centralized configuration system via `config/settings.py` and `.env` files for easy management of API keys and other parameters.
-   **Efficient Internal Data Handling**: Leverages Apache Feather format for internal data representation, significantly improving processing speed, memory efficiency, and data type preservation.

## Architecture

The system operates on a multi-agent paradigm, orchestrated by the `OrchestratorAgent`. The workflow is as follows:

1.  **User Input**: The user provides paths to raw CSV datasets, a meta-information file (describing data and analytical goals), and desired output paths.
2.  **`MetaAnalyzerAgent`**: Reads the meta-information file (e.g., `meta_output.txt`), uses an LLM to parse the desired schema, analytical questions, and a preliminary processing plan (`pipeline_plan` and `suggested_operations`). It then loads the raw CSV datasets, **converts them internally to the highly efficient Feather format**, and performs initial standardization (column renaming, type enforcement) and filtering based on the parsed meta-info. It also ensures data uniqueness after loading.
3.  **`OrchestratorAgent`**: Takes the prepared DataFrame and the LLM-generated processing plan. It sequentially delegates tasks to specialized agents for cleaning, modification, and transformation. It manages adaptive retries and incorporates learned optimizations from validation feedback.
4.  **Specialized Agents (`CleanerAgent`, `DataModifierAgent`, `TransformerAgent`)**: Each agent performs its specific set of operations on the data, leveraging adaptive logic to choose optimal methods based on data characteristics and LLM insights.
5.  **Validation Agents (`CleaningValidatorAgent`, `ModificationValidatorAgent`, `TransformationValidatorAgent`)**: After each processing stage, a dedicated validator agent assesses the quality of the work. If issues are found, it generates specific recommendations.
6.  **Feedback Loop**: The `OrchestratorAgent` receives validation results. If validation fails, it incorporates the validator's recommendations into the processing instructions and retries the failed stage (up to a configurable limit), making the pipeline adaptive.
7.  **`ProcessRecorderAgent`**: Continuously logs all activities, decisions, and outcomes throughout the entire pipeline, including the *reasons* for adaptive choices, creating a comprehensive audit trail.
8.  **Final Output**: Once all stages are successfully completed, the final purified DataFrame (which was processed internally as Feather) is saved to the specified output path, typically back into a `.csv` file.
9.  **Comprehensive Reporting**: The `ProcessRecorderAgent` generates a highly detailed JSON report summarizing the entire purification run, explaining *what* operations were performed, *why*, and their *impact* on data quality.

## Project Structure

```
<!-- The content of project_structure.txt will be inserted here by the user or a script -->
```

*Note: For a detailed explanation of each file and directory, please refer to `project_structure.txt`.*

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/data-purifier.git
    cd data-purifier
    ```

2.  **Create and activate a virtual environment**:
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On Unix or MacOS:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy and NLTK models**:
    The `MetaAnalyzerAgent` uses spaCy for NLP, and `TransformerAgent` uses NLTK. For consistent deployments, it's recommended to pre-download these models:
    ```bash
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
    ```

## Dockerization (Optional)

For consistent and isolated deployments, you can containerize the application using Docker.

1.  **Build the Docker image**:
    ```bash
    docker build -t data-purifier .
    ```
2.  **Run the Docker container**:
    ```bash
    docker run -e OPENAI_API_KEY="your_api_key" -v /host/path/to/data:/app/data data-purifier --dataset_paths "/app/data/yt.csv" --output_path "/app/data/Cleaned_yt.csv" --meta_output_path "/app/data/meta_output.txt"
    ```
    *   Replace `"your_api_key"` with your actual OpenAI API key.
    *   Map your local data directory to `/app/data` inside the container using `-v /host/path/to/data:/app/data`.
    *   Adjust `--dataset_paths`, `--output_path`, and `--meta_output_path` to reflect paths *inside the container* (e.g., `/app/data/your_file.csv`).

## Configuration

Configuration is managed via environment variables loaded from a `.env` file and settings defined in `config/settings.py`.

1.  **Create a `.env` file**: In the root directory of the project, create a file named `.env`.
2.  **Add API Keys**: Add your OpenAI API key (or other LLM provider keys) to this file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    # GOOGLE_API_KEY="your_google_api_key_here"
    # ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    ```
    *Note: If `OPENAI_API_KEY` is not set, the system will use a mock LLM for testing purposes, which will not provide real LLM capabilities.*

3.  **Review `config/settings.py`**: This file contains default settings and logic for loading environment variables. You can add or modify configuration variables here as needed.

## Usage

Run the main application from the project root directory. The script will interactively prompt you for the necessary file paths.

```bash
python main.py
```

**When prompted, provide the absolute paths for:**

1.  **Number of datasets**: Enter how many input CSV files you have.
2.  **Dataset paths**: For each dataset, enter its absolute path (e.g., `E:\Gemini CLI\Data_Purifier\yt.csv`).
3.  **Path to save processed dataset**: Enter the absolute path and desired filename for the final output CSV (e.g., `E:\Gemini CLI\Data_Purifier\Cleaned_yt.csv`). The system will create this file.
4.  **Path to meta output file**: Enter the absolute path to your meta-information file (e.g., `E:\Gemini CLI\Data_Purifier\meta_output.txt`). This file should contain your data schema, analytical questions, and initial processing suggestions.
5.  **Enable verbose logging?**: Type `yes` or `no`.

### Example `meta_output.txt` Content

Your `meta_output.txt` file should be structured in a way that the `MetaAnalyzerAgent` (via its LLM) can interpret it. Here's an example structure:

```text
schema:
  columns:
    - video_id
    - title
    - publish_time
    - views
    - likes
    - dislikes
    - comment_count
  rows: 100
  column_types:
    video_id: string
    title: string
    publish_time: datetime
    views: integer
    likes: integer
    dislikes: integer
    comment_count: integer
questions:
  - Analyze YouTube video performance.
pipeline_plan:
  - cleaning
  - modification
  - transformation
suggested_operations:
  cleaning_operations:
    video_id:
      - operation: remove_duplicates
        reason: Ensure unique video records.
  modification_operations: {}
  transformation_operations:
    publish_time:
      - operation: scale_normalize
        method: standard_scaler
        reason: Standardize publish time for analysis.
    views:
      - operation: scale_normalize
        method: standard_scaler
        reason: Standardize views for analysis.
    likes:
      - operation: scale_normalize
        method: standard_scaler
        reason: Standardize likes for analysis.
    dislikes:
      - operation: scale_normalize
        method: standard_scaler
        reason: Standardize dislikes for analysis.
    comment_count:
      - operation: scale_normalize
        method: standard_scaler
        reason: Standardize comment count for analysis.
```

## Internal Data Handling (Feather Files)

To optimize performance, the system internally uses the Apache Feather file format for data processing:

-   **Input**: You provide data in `.csv` format.
-   **Conversion**: The `MetaAnalyzerAgent` automatically converts input `.csv` files to `.feather` format upon loading and caches them. All subsequent processing by the `CleanerAgent`, `DataModifierAgent`, and `TransformerAgent` operates on this efficient binary format in memory.
-   **Benefits**: This approach significantly improves data loading/saving speed, reduces memory footprint, and ensures precise preservation of data types throughout the pipeline, minimizing potential data quality issues that can arise from repeated CSV parsing.
-   **Output**: The final purified dataset is saved back to your specified output path in `.csv` format for easy accessibility and compatibility.

## Running Tests

To ensure the system's functionality and stability, run the test suite using `pytest`:

```bash
pytest tests/
```

## Contributing

We welcome contributions to enhance this Data Purification System! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/your-feature-name`).
3.  Implement your changes, ensuring adherence to existing code style and conventions.
4.  Write and run tests to cover your new features or bug fixes.
5.  Commit your changes (`git commit -m 'Add your descriptive commit message'`).
6.  Push to your branch (`git push origin feature/your-feature-name`).
7.  Open a Pull Request to the main repository.

## Deployment Readiness

This system has been significantly enhanced for improved robustness, intelligence, and reliability, making it highly suitable for deployment, especially in single-developer environments. Key improvements include:

-   **Enhanced Error Handling & Logging**: Critical sections across agents now feature more specific error handling and detailed logging, providing clearer insights into potential issues during runtime and aiding in debugging.
-   **Refined Secrets Management Guidance**: The configuration loading (`config/settings.py`) now includes explicit recommendations for managing API keys and other sensitive information directly via environment variables in production, moving away from reliance on `.env` files for deployed systems.
-   **Version-Pinned Dependencies**: All project dependencies in `requirements.txt` are now explicitly version-pinned, ensuring reproducibility across different environments and preventing unexpected behavior due to library updates.
-   **Scalability & Performance Optimizations**: Implemented parallel data loading, optimized cleaning operations (e.g., adaptive imputation, robust duplicate removal), and efficient LLM interactions to enhance processing speed.
-   **Containerization**: A `Dockerfile` is provided, enabling consistent and isolated deployment across various environments.

While these enhancements significantly improve the system's stability and maintainability, further considerations for very large-scale production deployments might include:

-   **Advanced Monitoring & Alerting**: Integrating with external monitoring systems for real-time performance tracking and automated alerts.
-   **Distributed Processing**: For extremely large datasets, exploring distributed computing frameworks (e.g., Apache Spark) might be necessary.
-   **LLM Cost Optimization**: Implementing more aggressive token usage monitoring and potentially exploring local LLM solutions for certain tasks to reduce API costs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.