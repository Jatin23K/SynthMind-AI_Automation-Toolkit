# AI-Powered Data Analysis Tool

## Overview
This project is an intelligent data analysis tool that leverages AI agents to analyze and compare datasets. It uses OpenAI's language models through the CrewAI framework to provide deep insights into data structures, generate analytical questions, and compare multiple datasets.

## Key Features
- **Multi-Format Support**: Reads CSV, Excel, and JSON files
- **AI-Powered Analysis**: Uses three specialized AI agents:
  - Schema Sleuth: Analyzes data structure and metadata
  - Curious Catalyst: Generates analytical questions
  - Data Comparator: Compares multiple datasets
- **Comprehensive Analysis**: Provides detailed insights including:
  - Basic statistics (rows, columns)
  - Column descriptions and data types
  - Generated analytical questions
  - Dataset comparisons
- **Flexible Dataset Handling**: Can analyze one or multiple datasets simultaneously

## Requirements
- Python 3.x
- Required packages:
  - crewai
  - pandas
  - langchain
- OpenAI API key

## Installation
1. Clone the repository
2. Install required packages:
   ```bash
   pip install crewai pandas langchain
   ```
3. Set up your OpenAI API key:
   - Environment variable (recommended):
     - Windows: `set OPENAI_API_KEY=your-api-key-here`
     - Mac/Linux: `export OPENAI_API_KEY=your-api-key-here`
   - Or set directly in code (not recommended for security)

## Usage
1. Run the script
2. Enter the number of datasets you want to analyze
3. Provide the complete file path for each dataset
4. Review the analysis results:
   - Dataset summaries
   - Column descriptions
   - Generated questions
   - Dataset comparisons (if multiple datasets)

## Project Structure
- `Main script containing all functionality
- Dependencies are checked and installed automatically
- Error handling for file reading and analysis
- User-friendly interface for dataset input

## Features in Detail

### Data Reading
- Supports multiple file formats (CSV, Excel, JSON)
- Automatic format detection
- Error handling for invalid files

### Data Analysis
- Automatic column description generation
- Data type detection
- Statistical summaries
- Pattern recognition

### AI Agents
1. **Schema Sleuth**
   - Analyzes data structure
   - Creates metadata
   - Identifies patterns

2. **Curious Catalyst**
   - Generates analytical questions
   - Identifies potential insights
   - Suggests analysis directions

3. **Data Comparator**
   - Compares multiple datasets
   - Identifies relationships
   - Highlights differences

### Output
- Detailed dataset summaries
- Column descriptions
- Generated analytical questions
- Comparative analysis (for multiple datasets)

