from crewai import Task

class ModificationValidatorTasks:
    def validate_modification_task(self, agent, processed_dataframe, original_dataframe):
        return Task(
            description="""
            Validate the modified dataframe against the original or previously processed dataframe.
            Checks to perform:
            - Ensure column names are standardized as per instructions.
            - Verify new features are correctly engineered.
            - Confirm categorical encoding is applied correctly.
            Log all validation checks and their results.
            """,
            expected_output="""
            A boolean indicating whether the modified data is valid.
            """,
            agent=agent
        )