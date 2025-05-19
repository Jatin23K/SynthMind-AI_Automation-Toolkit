from crewai import Task

class ModifierTasks:
    def modify_data_task(self, agent, dataframe, processing_instructions):
        return Task(
            description=f"""
            Modify the features of the provided dataframe based on processing instructions: {processing_instructions}.
            Operations include:
            - Standardizing column names (e.g., to snake_case).
            - Engineering new features from existing ones.
            - Encoding categorical variables.
            Log all modification operations performed.
            """,
            expected_output="""
            A modified pandas DataFrame and a boolean indicating success.
            """,
            agent=agent
        )