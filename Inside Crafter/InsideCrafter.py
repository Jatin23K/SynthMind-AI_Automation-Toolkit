import pandas as pd
import openai
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import asyncio
import aiofiles
from crewai import Agent, Task, Crew
from io import StringIO
import contextlib

# === Configure Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === OpenAI API Key ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Define Agents ===
analyzer_agent = Agent(
    role="Data Analyzer",
    goal="Analyze cleaned data using metadata questions with intelligent reasoning and visual suggestions",
    backstory="A senior analyst that understands context, writes analysis and visualization code, and explains insights clearly.",
    verbose=True
)

visualizer_agent = Agent(
    role="Visualizer Executor",
    goal="Execute visualization code provided by the analyzer to generate charts",
    backstory="A skilled visual executor that renders high-quality visuals using matplotlib and seaborn based on instructions from the analyzer.",
    verbose=True
)

# === Define Task ===
analyzer_task = Task(
    description="Analyze cleaned data using metadata questions and output: question, analysis code, visual code, output, insights, and visualization.",
    agent=analyzer_agent,
    expected_output="Markdown report with full analysis and visuals."
)

# === GPT-Powered Code & Insight Generator ===
async def gpt_generate_code_insight_async(question, columns):
    prompt = f"""
You are a senior data analyst.

You are given this question: \"{question}\"

Columns available in the dataframe: {columns}

Write:
1. Python Pandas code to answer the question
2. Python matplotlib/seaborn code for a visual if helpful
3. A short paragraph insight explaining what the code reveals

Use only code that works with a Pandas DataFrame named 'cleaned_df'.
Respond in this format:
<ANALYSIS_CODE>
...
</ANALYSIS_CODE>
<VISUAL_CODE>
...
</VISUAL_CODE>
<INSIGHT>
...
</INSIGHT>
"""
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    content = response['choices'][0]['message']['content']
    analysis_code = content.split("<ANALYSIS_CODE>")[1].split("</ANALYSIS_CODE>")[0].strip()
    visual_code = content.split("<VISUAL_CODE>")[1].split("</VISUAL_CODE>")[0].strip()
    insight = content.split("<INSIGHT>")[1].split("</INSIGHT>")[0].strip()
    return analysis_code, visual_code, insight

# === Visualization Executor ===
def execute_visual_code(code, i):
    try:
        exec_globals = {"plt": plt, "sns": sns, "pd": pd}
        exec_locals = {}
        filename = f"visual_q{i}_{uuid.uuid4().hex[:6]}.png"
        exec(code, exec_globals, exec_locals)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        logging.warning(f"Failed to generate visual for Q{i}: {e}")
        return None

# === Analyze a Single Question ===
async def analyze_question(i, question, cleaned_df, report_file):
    try:
        analysis_code, visual_code, insight = await gpt_generate_code_insight_async(question, list(cleaned_df.columns))

        # Execute analysis code
        local_vars = {"cleaned_df": cleaned_df.copy()}
        with contextlib.redirect_stdout(StringIO()) as f:
            exec(analysis_code, {}, local_vars)
            output_text = f.getvalue()

        # Execute visual code
        vis_path = execute_visual_code(visual_code, i)

        async with aiofiles.open(report_file, mode="a", encoding="utf-8") as report:
            await report.write(f"### Question {i}:\n{question}\n\n")
            await report.write(f"**Analysis Code:**\n```python\n{analysis_code}\n```\n\n")
            await report.write(f"**Output:**\n```
{output_text}\n```\n\n")
            await report.write(f"**Insight:**\n{insight}\n\n")
            if visual_code.strip():
                await report.write(f"**Visual Code:**\n```python\n{visual_code}\n```\n\n")
            if vis_path:
                await report.write(f"**Visualization:** ![]({vis_path})\n\n")
            await report.write("---\n\n")

        logging.info(f"Analyzed and visualized question {i}")

    except Exception as e:
        async with aiofiles.open(report_file, mode="a", encoding="utf-8") as report:
            await report.write(f"### Question {i}:\n{question}\n\n")
            await report.write(f"**Error:** {e}\n\n---\n\n")

# === Analyze All Questions Asynchronously ===
async def analyze_with_visuals_async(cleaned_df, metadata_text, report_path="analysis_report.md"):
    questions = [q.strip() for q in metadata_text.strip().split("\n") if q.strip()]
    tasks = [analyze_question(i + 1, q, cleaned_df, report_path) for i, q in enumerate(questions)]
    await asyncio.gather(*tasks)

# === Entry Function to Run Analyzer Crew ===
def run_analyzer_crew(cleaned_data_path, meta_output_path):
    try:
        cleaned_df = pd.read_csv(cleaned_data_path)
        with open(meta_output_path, "r") as f:
            metadata_text = f.read()
        asyncio.run(analyze_with_visuals_async(cleaned_df, metadata_text))
        logging.info("Analyzer crew execution completed. Report generated.")
    except Exception as e:
        logging.error(f"Analyzer crew failed: {e}")

# === Full Crew Setup ===
analyzer_crew = Crew(
    agents=[analyzer_agent, visualizer_agent],
    tasks=[analyzer_task]
)

# analyzer_crew.run()  # Optional orchestration
