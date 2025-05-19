import pandas as pd

class DataModifierAgent:
    def __init__(self):
        self.modification_logs = []

    def process(self, df, processing_instructions):
        """Process the dataframe according to the instructions"""
        try:
            if processing_instructions.get('create_full_name'):
                if 'first_name' in df.columns and 'last_name' in df.columns:
                    df['full_name'] = df['first_name'] + ' ' + df['last_name']
                    self.modification_logs.append("Created full_name column")
            
            if processing_instructions.get('combine_columns'):
                columns = processing_instructions['combine_columns']
                separator = processing_instructions.get('separator', ' ')
                if all(col in df.columns for col in columns):
                    new_col_name = '_'.join(columns)
                    df[new_col_name] = df[columns].astype(str).agg(separator.join, axis=1)
                    self.modification_logs.append(f"Combined columns {columns} into {new_col_name}")
            
            if processing_instructions.get('rename_columns'):
                rename_map = processing_instructions['rename_columns']
                valid_renames = {old: new for old, new in rename_map.items() if old in df.columns}
                if valid_renames:
                    df.rename(columns=valid_renames, inplace=True)
                    self.modification_logs.append(f"Renamed columns: {valid_renames}")
            
            if processing_instructions.get('drop_columns'):
                columns_to_drop = [col for col in processing_instructions['drop_columns'] if col in df.columns]
                if columns_to_drop:
                    df.drop(columns=columns_to_drop, inplace=True)
                    self.modification_logs.append(f"Dropped columns: {columns_to_drop}")
            
            return df, True
        except Exception as e:
            self.modification_logs.append(f"Modification failed: {str(e)}")
            return df, False