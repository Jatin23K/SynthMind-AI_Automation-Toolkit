import pandas as pd
from pandas_profiling import ProfileReport
from ..agents import (
    cleaner_agent,
    modifier_agent,
    transformer_agent,
    summarizer_agent
)
from ..utils.logging_utils import setup_logging, log_process

class DataProcessor:
    def __init__(self):
        self.logger = setup_logging()
        self.feedback_logs = []
        
    def process_dataset(self, dataset_path, meta_output_path):
        df = pd.read_csv(dataset_path)
        
        # Initialize agents
        self.cleaner = cleaner_agent.create_cleaner_agent()
        self.modifier = modifier_agent.create_modifier_agent()
        self.transformer = transformer_agent.create_transformer_agent()
        self.summarizer = summarizer_agent.create_summarizer_agent()
        
        # Process pipeline
        df = self.clean_dataset(df)
        df = self.modify_dataset(df)
        df = self.transform_dataset(df)
        self.generate_summary()
        
        return df