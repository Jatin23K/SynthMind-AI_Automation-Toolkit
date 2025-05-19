from crewai import Task

class TransformationValidatorTasks:
    def validate_transformation_task(self, agent, processed_dataframe, original_dataframe):
        return Task(
            description="""
            Validate the transformed dataframe.
            Checks to perform:
            - Ensure numeric features are scaled correctly as per instructions.
            - Verify mathematical transformations are applied accurately.
            Log all validation checks and their results.
            """,
            expected_output="""
            A boolean indicating whether the transformed data is valid.
            """,
            agent=agent
        )