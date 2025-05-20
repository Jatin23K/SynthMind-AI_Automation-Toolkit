from sklearn.feature_selection import SelectKBest, f_classif
import logging

class FeatureSelector:
    @staticmethod
    def select_features(df, target_column):
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support(indices=True)]
            logging.info(f"Selected features: {selected_features}")
            return df[selected_features]
        return df