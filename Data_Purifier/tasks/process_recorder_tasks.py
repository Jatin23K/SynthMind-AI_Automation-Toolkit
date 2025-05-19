from crewai import Task

class ProcessRecorderTasks:
    def record_process_task(self, agent, analysis_logs, cleaning_logs, modification_logs, transformation_logs):
        return Task(
            description="""
            Record and summarize all data processing steps performed by the agents.
            Compile logs from:
            - Meta Analyzer Agent
            - Cleaner Agent
            - Modifier Agent
            - Transformer Agent
            Generate a comprehensive report in Markdown format.
            """,
            expected_output="""
            A string containing the formatted Markdown report of the entire process.
            """,
            agent=agent
        )