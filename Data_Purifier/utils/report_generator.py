import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class ReportGenerator:
    def __init__(self):
        self.full_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_stages": {}
        }

    def add_stage_report(self, stage_name: str, report_data: Dict):
        """Adds a report for a specific stage of the data processing pipeline.
        """
        self.full_report["pipeline_stages"][stage_name] = report_data
        self.full_report["pipeline_stages"][stage_name]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_report(self, output_path: str):
        """Save the complete report to a file.
        """
        report_path = Path(output_path)

        # Save as JSON
        with open(report_path.with_suffix('.json'), 'w') as f:
            json.dump(self.full_report, f, indent=2)

        # Generate HTML report
        html_report = self._generate_html_report()
        with open(report_path.with_suffix('.html'), 'w') as f:
            f.write(html_report)

    def _generate_html_report(self) -> str:
        """Generate a user-friendly HTML report from the full_report.
        """
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

        for stage_name, stage_data in self.full_report['pipeline_stages'].items():
            html += f"""
                <div class="stage-section">
                    <h2>Stage: {stage_name.replace('_', ' ').title()}</h2>
                    <p><strong>Timestamp:</strong> {stage_data.get('timestamp', 'N/A')}</p>
                    <p><strong>Summary:</strong> <span class="status-{stage_data.get('status', 'unknown')}">{stage_data.get('summary', 'No summary provided.')}</span></p>
            """

            if "issues_found" in stage_data and stage_data["issues_found"]:
                html += "<h3>Issues Found:</h3><ul>"
                for issue_type, issue_details in stage_data["issues_found"].items():
                    html += f"<li><strong>{issue_type.replace('_', ' ').title()}:</strong> <pre>{json.dumps(issue_details, indent=2)}</pre></li>"
                html += "</ul>"

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

        html += """
            </div>
        </body>
        </html>
        """

        return html
