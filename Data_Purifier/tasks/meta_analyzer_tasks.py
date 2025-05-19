from crewai import Task

class MetaAnalyzerTasks:
    def analyze_metadata_task(self, agent, dataset_paths, meta_output_path):
        return Task(
            description=f"""
            Analyze the metadata from {meta_output_path} and validate the dataset paths: {dataset_paths}.
            Extract processing instructions from the metadata to guide subsequent agents.
            Ensure all dataset paths are valid and accessible.
            Log all analysis steps and findings.
            """,
            expected_output="""
            A dictionary containing:
            - 'valid_dataset_paths': A list of validated and accessible dataset file paths.
            - 'processing_instructions': A dictionary with instructions for cleaning, modification, and transformation.
            - 'metadata_content': The content of the metadata file.
            """,
            agent=agent
        )