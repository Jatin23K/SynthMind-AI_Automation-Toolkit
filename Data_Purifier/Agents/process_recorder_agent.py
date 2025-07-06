# agents/process_recorder_agent.py
# This agent is dedicated to logging and recording all significant activities
# and outcomes throughout the data purification pipeline. It serves as an audit trail
# and generates a comprehensive final report.

import json
import logging
from datetime import datetime
from typing import Dict


class ProcessRecorderAgent:
    """
    The ProcessRecorderAgent is responsible for maintaining a detailed log
    of all tasks, statuses, and outcomes during the data purification process.
    It aggregates this information to generate a final, comprehensive report.
    """

    def __init__(self):
        """
        Initializes the ProcessRecorderAgent.
        """
        self.logger = logging.getLogger(__name__)
        self.pipeline_activities = [] # List to store chronological records of all activities
        self.stage_reports = {} # Dictionary to store summary reports from each pipeline stage

    def record_task_activity(self, agent_name: str, task_description: str, status: str, details: Dict = None, reason: str = None):
        """
        Records a specific task activity performed by an agent within the pipeline.

        Args:
            agent_name (str): The name of the agent performing the task.
            task_description (str): A brief description of the task being performed.
            status (str): The current status of the task (e.g., "Initiated", "In Progress", "Success", "Failed").
            details (Dict, optional): An optional dictionary containing additional context or results for the task.
        """
        activity_entry = {
            "timestamp": datetime.now().isoformat(), # ISO formatted timestamp for when the activity occurred
            "agent_name": agent_name, # Name of the agent
            "task_description": task_description, # Description of the task
            "status": status, # Status of the task
            "details": details if details is not None else {}, # Additional details
            "reason": reason # Reason for the operation, if applicable
        }
        self.pipeline_activities.append(activity_entry) # Add the activity to the list
        self.logger.info(f"[{agent_name}] {task_description} - {status}") # Log the activity

    def add_stage_report(self, stage_name: str, report_data: Dict):
        """
        Adds a comprehensive report for a specific stage of the pipeline.
        This allows for detailed summaries of each major processing step.

        Args:
            stage_name (str): The name of the pipeline stage (e.g., "cleaning", "modification").
            report_data (Dict): A dictionary containing the summary report for that stage.
        """
        self.stage_reports[stage_name] = report_data # Store the stage report
        self.logger.info(f"Report added for stage: {stage_name}")

    def generate_final_report(self, output_path: str) -> Dict:
        """
        Generates and saves the final consolidated report for the entire pipeline run.
        This report includes all recorded activities and stage summaries.

        Args:
            output_path (str): The base path where the final JSON report will be saved.
                                The report will be saved as `{output_path}.json`.

        Returns:
            Dict: The complete final report as a dictionary.
        """
        final_report = {
            "pipeline_run_timestamp": datetime.now().isoformat(), # Timestamp of the report generation
            "overall_status": "Completed", # Default status, updated by Orchestrator based on pipeline outcome
            "summary_overview": self._generate_summary_overview(), # New: High-level summary
            "pipeline_activities": self.pipeline_activities, # All recorded activities
            "stage_summaries": self.stage_reports # Summaries from each stage
        }

        try:
            # Save the final report as a JSON file
            with open(f"{output_path}.json", 'w') as f:
                json.dump(final_report, f, indent=2) # Pretty print JSON with 2-space indentation
            self.logger.info(f"Final JSON report saved to {output_path}.json")
        except Exception as e:
            self.logger.error(f"Error saving final JSON report: {e}", exc_info=True)

        return final_report

    def _generate_summary_overview(self) -> Dict:
        """
        Generates a high-level, human-readable summary of the entire pipeline run,
        with as much detail as possible.
        """
        summary = {
            "overall_pipeline_status": "",
            "pipeline_duration_seconds": 0,
            "initial_data_state": "Not captured in detail by recorder (captured by MetaAnalyzerAgent)",
            "final_data_state": "Not captured in detail by recorder (captured by OrchestratorAgent)",
            "detailed_operations_log": {
                "cleaning_stage": [],
                "modification_stage": [],
                "transformation_stage": []
            },
            "data_quality_impact": {
                "duplicates_removed_total": 0,
                "missing_values_imputed_total": 0,
                "outliers_handled_total": 0,
                "inconsistencies_resolved_total": 0,
                "new_features_created_total": 0,
                "columns_removed_total": 0
            },
            "stage_validation_results": {}
        }

        start_time = None
        end_time = None

        for activity in self.pipeline_activities:
            timestamp = datetime.fromisoformat(activity["timestamp"])

            if activity["agent_name"] == "OrchestratorAgent":
                if "Starting data purification pipeline" in activity["task_description"]:
                    start_time = timestamp
                    # Temporarily commented out due to f-string SyntaxError
                    # summary["initial_data_state"] = f"Pipeline initiated. Input datasets: {', '.join(activity["details"].get('dataset_paths', ['N/A']))}"
                elif "Data purification pipeline finished." in activity["task_description"]:
                    end_time = timestamp
                    summary["overall_pipeline_status"] = activity["status"]
                    summary["final_data_state"] = "Final DataFrame shape: {}".format(activity["details"].get('final_shape', 'N/A'))

            # Process cleaning operations
            if activity["agent_name"] == "CleanerAgent" and activity["status"] == "Success":
                op_details = activity["details"]
                if "operations_performed_details" in op_details:
                    for op in op_details["operations_performed_details"]:
                        log_entry = {
                            "operation": op["operation"],
                            "column": op["column"],
                            "status": op["status"],
                            "details": op["details"],
                            "reason": activity.get("reason", "N/A") # Use reason from activity if available
                        }
                        summary["detailed_operations_log"]["cleaning_stage"].append(log_entry)

                        # Update data quality impact metrics
                        if op["operation"] == "remove_duplicates" and "Removed" in op["details"]:
                            match = re.search(r'Removed (\d+) duplicate rows', op["details"])
                            if match: summary["data_quality_impact"]["duplicates_removed_total"] += int(match.group(1))
                        elif op["operation"] == "handle_missing_values" and "Handled missing values" in op["details"]:
                            # This is tricky as details might not contain count directly. Rely on validation report for total.
                            pass
                        elif op["operation"] == "handle_outliers" and ("Clipped" in op["details"] or "Removed" in op["details"]):
                            match = re.search(r'(Clipped|Removed) (\d+) outliers', op["details"])
                            if match: summary["data_quality_impact"]["outliers_handled_total"] += int(match.group(2))
                        elif op["operation"] == "handle_inconsistencies" and "Handled inconsistencies" in op["details"]:
                            match = re.search(r'mapping (\d+) values', op["details"])
                            if match: summary["data_quality_impact"]["inconsistencies_resolved_total"] += int(match.group(1))

            # Process modification operations
            if activity["agent_name"] == "DataModifierAgent" and activity["status"] == "Success":
                op_details = activity["details"]
                if "operations_performed" in op_details:
                    for op in op_details["operations_performed"]:
                        log_entry = {
                            "operation": op["operation"],
                            "column": op["column"],
                            "status": op["status"],
                            "details": op["details"],
                            "reason": activity.get("reason", "N/A")
                        }
                        summary["detailed_operations_log"]["modification_stage"].append(log_entry)
                        if op["operation"] == 'create_new_feature':
                            summary["data_quality_impact"]["new_features_created_total"] += 1
                        elif op["operation"] == 'drop_column':
                            summary["data_quality_impact"]["columns_removed_total"] += 1

            # Process transformation operations
            if activity["agent_name"] == "TransformerAgent" and activity["status"] == "Success":
                op_details = activity["details"]
                if "operations_performed" in op_details:
                    for op in op_details["operations_performed"]:
                        log_entry = {
                            "operation": op["operation"],
                            "column": op["column"],
                            "status": op["status"],
                            "details": op["details"],
                            "reason": activity.get("reason", "N/A")
                        }
                        summary["detailed_operations_log"]["transformation_stage"].append(log_entry)

            # Capture validation results
            if "validation passed" in activity["task_description"].lower() or "validation failed" in activity["task_description"].lower():
                stage_name = "unknown"
                if "cleaning validation" in activity["task_description"].lower(): stage_name = "cleaning"
                elif "modification validation" in activity["task_description"].lower(): stage_name = "modification"
                elif "transformation validation" in activity["task_description"].lower(): stage_name = "transformation"

                if stage_name != "unknown":
                    summary["stage_validation_results"][stage_name] = {
                        "status": activity["status"],
                        "report": activity["details"]
                    }
                    # Update missing values imputed from cleaning validation report
                    if stage_name == "cleaning" and "issues_found" in activity["details"] and "missing_values" in activity["details"]["issues_found"]:
                        for col, count in activity["details"]["issues_found"]["missing_values"].items():
                            summary["data_quality_impact"]["missing_values_imputed_total"] += count

        if start_time and end_time:
            summary["pipeline_duration_seconds"] = (end_time - start_time).total_seconds()

        # Add stage summaries from self.stage_reports (already populated by add_stage_report)
        for stage, report_data in self.stage_reports.items():
            if stage not in summary["stage_validation_results"]:
                summary["stage_validation_results"][stage] = report_data
            else:
                # Merge if validation results already exist
                summary["stage_validation_results"][stage].update(report_data)

        return summary