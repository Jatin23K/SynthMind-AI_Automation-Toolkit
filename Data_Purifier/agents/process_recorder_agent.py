from crewai import Agent
from langchain_openai import ChatOpenAI # Or your preferred LLM
import os
# Assuming file_utils is in the parent directory's utils folder
from ..utils.file_utils import save_json_report, save_dataframe_as_csv # Removed save_process_report if not used for text

class ProcessRecorderAgent:
    def __init__(self, llm=None):
        self.llm = llm if llm else ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo") # Adjusted for potentially less complex task
        self.agent = Agent(
            role='Process Documentation Specialist',
            goal='To meticulously record all data processing steps, changes, reasons, and dataset states, and to compile comprehensive reports in specified formats.',
            backstory=(
                "An expert in process documentation and data auditing, this agent ensures every step of the data journey is traceable "
                "and understandable. It transforms raw logs into structured, insightful reports and ensures final data products are accessible."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )
        self.report_data = [] # To store structured log data for JSON report

    def _format_log_entry(self, agent_name, task_description, status, details=None):
        """Helper to create a structured log entry."""
        return {
            "agent_name": agent_name,
            "task_description": task_description,
            "status": status,
            "details": details
        }

    def record_task_activity(self, agent_name, task_description, status, details=None):
        """Records a single task's activity. Called by other agents or the main pipeline."""
        entry = self._format_log_entry(agent_name, task_description, status, details)
        self.report_data.append(entry)
        print(f"[Log Recorded by {agent_name}]: {task_description} - {status}") # Simple console log for immediate feedback

    def generate_json_report(self, output_dir, filename_prefix=""):
        """Generates and saves the JSON report."""
        if not self.report_data:
            print("No data to generate JSON report.")
            return None
        
        formatted_report = {}
        for entry in self.report_data:
            agent_name_key = f"----- {entry['agent_name']} -----"
            if agent_name_key not in formatted_report:
                formatted_report[agent_name_key] = []
            
            task_entry = {
                "Task Description": entry['task_description'],
                "Status": entry['status'],
                "Details": entry['details']
            }
            formatted_report[agent_name_key].append(task_entry)

        report_filename = f"{filename_prefix}processing_details.json" if filename_prefix else "processing_details.json"
        return save_json_report(formatted_report, output_dir, report_filename)

    def save_cleaned_dataset(self, dataframe, output_dir, filename_prefix=""):
        """Saves the final cleaned DataFrame as a CSV file."""
        if dataframe is None:
            print("No DataFrame provided to save as CSV.")
            return None
        csv_filename = f"{filename_prefix}cleaned_dataset.csv" if filename_prefix else "cleaned_dataset.csv"
        return save_dataframe_as_csv(dataframe, output_dir, csv_filename)

    def finalize_reports(self, final_df, output_dir, dataset_name=""):
        """Main method to generate all reports and save the dataset."""
        filename_prefix = f"{dataset_name.replace('.csv', '').replace('.xlsx', '')}_" if dataset_name else ""
        
        json_report_path = self.generate_json_report(output_dir, filename_prefix)
        csv_path = self.save_cleaned_dataset(final_df, output_dir, filename_prefix)
        
        # Clear data for next dataset if any. Important if agent instance is reused.
        # self.report_data = [] 
        # Decided to keep logs for the lifetime of the agent instance, 
        # or clear them explicitly in main.py if a single recorder is used for multiple datasets sequentially.
        # For now, let's assume one recorder per dataset processing run or it's cleared outside.

        return json_report_path, csv_path

    def clear_logs(self): # Added a method to clear logs if needed
        self.report_data = []
        print("ProcessRecorderAgent logs cleared.")