import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTransformerAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.transformation_logs = []
    
    def process(self, df, processing_instructions):
        """Process the dataframe according to the instructions"""
        try:
            if processing_instructions.get('scale_numeric'):
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
                    self.transformation_logs.append(f"Scaled numeric columns: {list(numeric_cols)}")
            
            if processing_instructions.get('encode_categorical'):
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if col in df.columns:
                        df[f'{col}_encoded'] = pd.factorize(df[col])[0]
                        self.transformation_logs.append(f"Encoded categorical column: {col}")
            
            if processing_instructions.get('fill_missing'):
                strategy = processing_instructions.get('fill_strategy', 'mean')
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if strategy == 'mean':
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif strategy == 'median':
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif strategy == 'mode':
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
                self.transformation_logs.append(f"Filled missing values using {strategy} strategy")
            
            if processing_instructions.get('bin_numeric'):
                bin_cols = processing_instructions['bin_numeric']
                n_bins = processing_instructions.get('n_bins', 5)
                for col in bin_cols:
                    if col in df.columns and df[col].dtype in ['int64', 'float64']:
                        df[f'{col}_binned'] = pd.qcut(df[col], n_bins, labels=False)
                        self.transformation_logs.append(f"Binned numeric column {col} into {n_bins} bins")
            
            return df, True
        except Exception as e:
            self.transformation_logs.append(f"Transformation failed: {str(e)}")
            return df, False