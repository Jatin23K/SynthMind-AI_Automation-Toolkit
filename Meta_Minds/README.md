# 🧠 Meta_Minds (Agent-DS)

Meta_Minds is an AI-powered, modular data analysis framework that automates the complete data pipeline — from loading and cleaning to analyzing and generating insights using OpenAI's API.

This project is part of the **Agent-DS** system, designed to behave like an intelligent data assistant with capabilities to:
- Automatically detect data types and structures
- Clean, preprocess, and transform datasets
- Analyze patterns and extract insights using AI
- Prepare data for reporting or visualization
- Log and handle outputs systematically

---

## 📁 Project Structure

data_purifier/
├── .env   # This file is typically used to store environment variables, such as API keys
├── .gitignore  # Specifies intentionally untracked files that Git should ignore, such as compiled Python files ( __pycache__ ), virtual environment folders ( venv ), and potentially output files ( meta_output.txt ).
├── agents.py  # Contains the definitions for the CrewAI agents used in the project. These agents are configured with specific roles, goals, and backstories to perform tasks.
├── api.py  # code for setting up a web API, potentially using a framework like FastAPI (indicated by fastapi.exe and uvicorn.exe in the venv/Scripts directory), to interact with the data purification process programmatically.
├── config.py  # Handles application-wide configuration, including setting up logging and initializing the OpenAI client using the API key from the environment variables.
├── data_analyzer.py # Contains functions for analyzing the loaded data, such as generating summaries or descriptions of the datasets and their columns, likely utilizing the OpenAI client.
├── data_loader.py  #Responsible for loading data from various file formats (CSV, XLSX, JSON) into pandas DataFrames.
├── main.py  # The main entry point of the application. It orchestrates the entire workflow, from getting user input for file paths to processing data, creating agents and tasks, running the CrewAI, and saving the output.
├── meta_output.txt # The final output
├── output_handler.py  # Responsible for saving the final output of the data analysis process.
├── project_structure.txt
├── requirements.txt  # Lists the Python dependencies required for the project to run.
└─ tasks.py  # # Defines the CrewAI tasks that the agents will perform, such as analyzing data, generating questions, or comparing datasets.

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Jatin23K/Agent-DS.git
cd Agent-DS
````

2. **Create a Virtual Environment**

```bash
python -m venv venv
```

3. **Activate the Environment**

* On Windows:

```bash
venv\Scripts\activate
```

* On macOS/Linux:

```bash
source venv/bin/activate
```

4. **Install Dependencies**

```bash
pip install -r requirements.txt
```

5. **Configure OpenAI API Key**

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 How to Run

To run the data analysis crew, execute the main script (assuming `main.py` is the entry point):

```bash
python main.py
```

The script will load the datasets, create the agents and tasks, and run the CrewAI process to generate questions.


## 🧠 Technologies Used

* **Python 3.x**
* **OpenAI API (GPT-4 / GPT-3.5)**
* **dotenv**
* **pandas**
* **logging**
* **modular architecture**

---

## 🔐 Security

> ⚠️ Do **NOT** share your OpenAI API key.
> Ensure `.env` is included in your `.gitignore`.

---

## 🧑‍💻 Author

**Jatin Kumar**
📍 Gurgaon, India
📧 [jatinkumar20802@gmail.com]

---

