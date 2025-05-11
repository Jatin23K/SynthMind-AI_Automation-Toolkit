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

Agent-DS/
│
├── agents.py            # Agent-based orchestration logic
├── config.py            # Loads environment variables & sets up OpenAI client
├── data\_analyzer.py     # Performs intelligent data analysis
├── data\_loader.py       # Loads and prepares datasets
├── main.py              # Entry point for running the full pipeline
├── output\_handler.py    # Manages and logs output results
├── tasks.py             # Defines automation tasks or pipelines
├── requirements.txt     # Python dependencies
├── .gitignore           # Files and folders excluded from Git tracking
└── README.md            # Project documentation

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

Run the full system via:

```bash
python main.py
```

Or test components individually:

```bash
python data_loader.py
python data_analyzer.py
```

---

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

## 📌 Use Cases

* Automating repetitive data analysis tasks
* Generating AI-powered insights
* Building intelligent ETL pipelines
* Modular AI agent design using scripts

---

## 🧑‍💻 Author

**Jatin Kumar**
📍 Gurgaon, India
📧 [jatinkumar20802@gmail.com](mailto:jatinkumar20802@gmail.com)

---

