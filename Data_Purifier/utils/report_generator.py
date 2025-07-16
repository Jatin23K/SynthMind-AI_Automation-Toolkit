import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class ReportGenerator:
    """
    Generates comprehensive reports for the data purification pipeline.
    It collects information about different pipeline stages and can save them
    in JSON and user-friendly HTML formats.
    """
    def __init__(self):
        """
        Initializes the ReportGenerator.
        Sets up the basic structure of the full report, including a timestamp
        and an empty dictionary to store reports from various pipeline stages.
        """
        self.full_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Timestamp of when the report is initialized.
            "pipeline_stages": {} # Dictionary to hold reports for each stage of the pipeline.
        }

    def add_stage_report(self, stage_name: str, report_data: Dict):
        """
        Adds a comprehensive report for a specific stage of the data processing pipeline.
        This method allows accumulating reports from different agents or stages into a single, consolidated report.

        Args:
            stage_name (str): The name of the pipeline stage (e.g., "cleaning", "modification", "transformation").
            report_data (Dict): A dictionary containing the summary report for that stage.
                                 This typically includes status, issues found, and operations performed.
        """
        self.full_report["pipeline_stages"][stage_name] = report_data # Store the report data under the stage name.
        # Add a timestamp to the specific stage report, indicating when it was added.
        self.full_report["pipeline_stages"][stage_name]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_report(self, output_path: str):
        """
        Saves the complete consolidated report to files.
        It generates both a JSON file (for machine readability) and an HTML file (for human readability).

        Args:
            output_path (str): The base path for saving the report files.
                                The JSON report will be saved as `{output_path}.json`
                                and the HTML report as `{output_path}.html`.
        """
        report_path = Path(output_path) # Convert the output path string to a Path object for easier manipulation.

        # Save the full report as a JSON file.
        with open(report_path.with_suffix('.json'), 'w') as f:
            json.dump(self.full_report, f, indent=2) # Write the JSON data with a 2-space indentation for readability.

        # Generate the HTML report content.
        html_report = self._generate_html_report()
        # Save the generated HTML content to an HTML file.
        with open(report_path.with_suffix('.html'), 'w') as f:
            f.write(html_report)

    def _generate_html_report(self) -> str:
        """
        Generates a user-friendly HTML report from the collected full_report data.
        This method structures the report with styling for better readability in a web browser.

        Returns:
            str: A string containing the complete HTML content of the report.
        """
        # Start building the HTML string with basic HTML structure and CSS styling.
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Processing Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }}
                .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #0056b3; }}
                .stage-section {{ border-bottom: 1px solid #eee; padding-bottom: 20px; margin-bottom: 20px; }}
                .stage-section:last-child {{ border-bottom: none; margin-bottom: 0; }}
                .operation-item {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
                pre {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .status-success {{ color: green; font-weight: bold; }}
                .status-failed {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Data Processing Pipeline Report</h1>
                <p>Generated on: {self.full_report['timestamp']}</p>
        """

        # Iterate through each pipeline stage and append its details to the HTML string.
        for stage_name, stage_data in self.full_report['pipeline_stages'].items():
            html += f"""
                <div class="stage-section">
                    <h2>Stage: {stage_name.replace('_', ' ').title()}</h2>
                    <p><strong>Timestamp:</strong> {stage_data.get('timestamp', 'N/A')}</p>
                    <p><strong>Summary:</strong> <span class="status-{stage_data.get('status', 'unknown')}">{stage_data.get('summary', 'No summary provided.')}</span></p>
            """

            # If there are issues found in this stage, add them to the HTML.
            if "issues_found" in stage_data and stage_data["issues_found"]:
                html += "<h3>Issues Found:</h3><ul>"
                for issue_type, issue_details in stage_data["issues_found"].items():
                    html += f"<li><strong>{issue_type.replace('_', ' ').title()}:</strong> <pre>{json.dumps(issue_details, indent=2)}</pre></li>"
                html += "</ul>"

            # If there are operations performed in this stage, add their details to the HTML.
            if "operations_performed" in stage_data and stage_data["operations_performed"]:
                html += "<h3>Operations Performed:</h3>"
                for op in stage_data["operations_performed"]:
                    html += f"""
                        <div class="operation-item">
                            <h4>Operation: {op.get('operation', 'N/A')}</h4>
                            <p><strong>Column:</strong> {op.get('column', 'N/A')}</p>
                            <p><strong>Method:</strong> {op.get('method', 'N/A')}</p>
                            <p><strong>Status:</strong> <span class="status-{op.get('status', 'unknown')}">{op.get('status', 'N/A')}</span></p>
                            <p><strong>Details:</strong> <pre>{json.dumps(op.get('details', {}), indent=2)}</pre></p>
                        </div>
                    """
            html += "</div>"

        # Close the main container and body tags.
        html += """
            </div>
        </body>
        </html>
        """

        return html