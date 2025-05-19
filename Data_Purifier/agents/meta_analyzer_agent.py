from crewai import Agent
import pandas as pd
import os
import json

class MetaAnalyzerAgent:
    def __init__(self):
        self.agent = Agent(
            role="Meta Analyzer",
            goal="Analyze metadata and prepare datasets for processing",
            backstory="Expert in interpreting metadata and preparing datasets for optimal processing",
            verbose=True
        )
        self.analysis_logs = []
    
    def analyze(self, dataset_paths, meta_output_path):
        try:
            # Load metadata
            if os.path.exists(meta_output_path):
                with open(meta_output_path, 'r') as f:
                    metadata = f.read()
                self.log_analysis(f"Loaded metadata from {meta_output_path}")
            else:
                metadata = ""
                self.log_analysis(f"Warning: Metadata file {meta_output_path} not found")
            
            # Analyze metadata and prepare processing instructions
            processing_instructions = self.extract_processing_instructions(metadata)
            
            # Validate dataset paths
            valid_paths = []
            for path in dataset_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                    self.log_analysis(f"Validated dataset path: {path}")
                else:
                    self.log_analysis(f"Warning: Dataset path not found: {path}")
            
            # Load and filter the dataset based on required columns
            filtered_df = None
            if valid_paths:
                try:
                    # Assuming the first valid path is the primary dataset to analyze
                    df = pd.read_csv(valid_paths[0])
                    required_cols = processing_instructions.get('required_columns', [])
                    if required_cols and all(col in df.columns for col in required_cols):
                        filtered_df = df[required_cols]
                        self.log_analysis(f"Filtered dataset {valid_paths[0]} to required columns: {required_cols}")
                    elif required_cols:
                         self.log_analysis(f"Warning: Not all required columns {required_cols} found in dataset {valid_paths[0]}. Returning full dataset.")
                         filtered_df = df # Return full dataset if required columns are not all present
                    else:
                        self.log_analysis(f"No required columns specified in metadata for {valid_paths[0]}. Returning full dataset.")
                        filtered_df = df # Return full dataset if no required columns are specified
                except Exception as load_e:
                    self.log_analysis(f"Error loading or filtering dataset {valid_paths[0]}: {str(load_e)}")
                    # If loading/filtering fails, proceed without a filtered dataframe
                    filtered_df = None

            return {
                "valid_dataset_paths": valid_paths,
                "processing_instructions": processing_instructions,
                "metadata_content": metadata,
                "filtered_dataset": filtered_df # Include the filtered dataset in the results
            }, True
        except Exception as e:
            self.log_analysis(f"Meta analysis failed: {str(e)}")
            return {
                "valid_dataset_paths": dataset_paths,
                "processing_instructions": {},
                "metadata_content": "",
                "filtered_dataset": None # Return None for filtered dataset on failure
            }, False
    
    def extract_processing_instructions(self, metadata):
        # Extract processing instructions from metadata
        # This is a placeholder implementation - customize based on your metadata format
        instructions = {
            "cleaning": {
                "handle_missing": True,
                "remove_duplicates": True,
                "standardize_formats": True
            },
            "modification": {
                "rename_columns": True,
                "engineer_features": True
            },
            "transformation": {
                "scale_numeric": True,
                "encode_categorical": True
            }
        }
        
        # Parse metadata to update instructions
        # Example: if "no_scaling" is in metadata, set scale_numeric to False
        if "no_scaling" in metadata.lower():
            instructions["transformation"]["scale_numeric"] = False
        
        # Add more parsing logic based on your metadata format
        # Example: Identify required columns based on questions in metadata
        required_columns = []
        # This is a placeholder - customize based on your metadata format and how columns are referenced in questions
        for line in metadata.splitlines():
            if line.strip().lower().startswith("question:"):
                # Simple heuristic: try to find words that look like column names after 'Question:'
                # This needs to be replaced with actual parsing logic based on metadata structure
                question_text = line.split(":", 1)[1].strip()
                # Example: Extract words that might be column names (very basic)
                potential_columns = [word.strip('?.,!').lower() for word in question_text.split()]
                # Add potential columns to the list (avoiding duplicates and common words)
                for col in potential_columns:
                    if col and col not in required_columns and col not in ['what', 'is', 'the', 'user\'s', 'age', 'id', 'name']:# Add more common words to ignore
                         required_columns.append(col)

        instructions['required_columns'] = required_columns

        return instructions
    
    def log_analysis(self, message):
        self.analysis_logs.append(message)