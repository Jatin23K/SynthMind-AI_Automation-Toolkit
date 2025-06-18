# Meta Minds - AI-Powered Data Analysis Tool

![Meta Minds Logo](https://via.placeholder.com/150)  <!-- Replace with your actual logo -->

## 📝 Overview
Meta Minds is an intelligent data analysis tool that leverages AI to automatically generate insightful questions and analyses from your datasets. It's designed to help data analysts and researchers quickly understand their data and generate meaningful analytical questions.

## ✨ Features

- **Multi-format Support**: Works with CSV, Excel, and JSON files
- **AI-Powered Analysis**: Uses advanced AI to understand your data
- **Automated Question Generation**: Generates relevant analytical questions
- **Comparative Analysis**: Compares multiple datasets to find insights
- **Detailed Summaries**: Provides comprehensive data summaries and statistics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/meta-minds.git
   cd meta-minds
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 🛠️ Usage

1. **Run the application**
   ```bash
   python main.py
   ```

2. **Follow the prompts**
   - Enter the number of datasets you want to analyze
   - Provide the full paths to your dataset files
   - View the generated analysis in the console and in `meta_output.txt`

## 📂 Project Structure

```
meta-minds/
├── .env                    # Environment variables
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── main.py                 # Main application entry point
├── config.py               # Configuration settings
├── data_loader.py          # Data loading utilities
├── data_analyzer.py        # Data analysis functions
├── agents.py               # AI agent definitions
├── tasks.py                # Task definitions
└── output_handler.py       # Output management
```

## 🤖 AI Agents

### Schema Sleuth
- Analyzes data structure and schema
- Identifies data types and patterns
- Provides high-level dataset overview

### Curious Catalyst
- Generates insightful analytical questions
- Identifies trends and anomalies
- Suggests potential areas for deeper analysis

## 📊 Example Output

```
--- Dataset: sales_data.csv ---
• Rows: 10,000
• Columns: 15
• Analysis complete

--- Questions for sales_data.csv ---
1. What is the correlation between marketing spend and sales revenue?
2. Which product category has the highest profit margin?
3. How do sales vary by region and season?
...
```
