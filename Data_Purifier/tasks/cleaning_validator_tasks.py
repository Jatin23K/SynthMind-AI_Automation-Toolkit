from crewai import Task

class CleaningValidatorTasks:
    def validate_cleaning_task(self, agent, processed_dataframe, original_dataframe):
        return Task(
            description="""
            Validate the cleaned dataframe against the original dataframe.
            Checks to perform:
            - Ensure missing values are handled appropriately.
            - Confirm no duplicate rows exist (if specified by instructions).
            - Verify format consistency (e.g., date formats, numeric types).
            - Check overall data integrity.
            Log all validation checks and their results.
            """,
            expected_output="""
            A boolean indicating whether the cleaned data is valid.
            """,
            agent=agent
        )