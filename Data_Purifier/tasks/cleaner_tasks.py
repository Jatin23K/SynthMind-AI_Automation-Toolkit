from crewai import Task

class CleanerTasks:
    def clean_data_task(self, agent, dataframe, processing_instructions):
        return Task(
            description=f"""
            Clean the provided dataframe based on the following processing instructions: {processing_instructions}.
            Operations include:
            - Removing duplicate rows.
            - Handling missing values (e.g., imputation, removal).
            - Standardizing data formats (e.g., dates, numbers).
            Log all cleaning operations performed.
            """,
            expected_output="""
            A cleaned pandas DataFrame and a boolean indicating success.
            """,
            agent=agent,
            context=[dataframe]  # Pass dataframe in context if needed by the agent's tool
        )