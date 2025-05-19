from crewai import Task

class TransformerTasks:
    def transform_data_task(self, agent, dataframe, processing_instructions):
        return Task(
            description=f"""
            Apply data transformations to the provided dataframe based on processing instructions: {processing_instructions}.
            Operations include:
            - Scaling numeric features (e.g., normalization, standardization).
            - Applying mathematical transformations (e.g., log, power).
            Log all transformation operations performed.
            """,
            expected_output="""
            A transformed pandas DataFrame and a boolean indicating success.
            """,
            agent=agent
        )