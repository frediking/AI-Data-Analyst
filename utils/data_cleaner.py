import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log = []

    def clean_numeric(self, column, method, value=None):
        if method == "Mean":
            self.df[column] = self.df[column].fillna(self.df[column].mean())
        elif method == "Median":
            self.df[column] = self.df[column].fillna(self.df[column].median())
        elif method == "Mode":
            self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        elif method == "Constant Value":
            self.df[column] = self.df[column].fillna(value)
        elif method == "Interpolate":
            self.df[column] = self.df[column].interpolate()
        elif method == "Forward Fill":
            self.df[column] = self.df[column].ffill()
        elif method == "Back Fill":
            self.df[column] = self.df[column].bfill()
        elif method == "Drop Rows":
            self.df = self.df.dropna(subset=[column])
        else:
            raise ValueError(f"Unknown numeric cleaning method: {method}")
        self.cleaning_log.append((column, method, value))

    def clean_categorical(self, column, method, value=None):
        if method == "Mode":
            self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        elif method == "Constant Value":
            self.df[column] = self.df[column].fillna(value)
        elif method == "Drop Rows":
            self.df = self.df.dropna(subset=[column])
        else:
            raise ValueError(f"Unknown categorical cleaning method: {method}")
        self.cleaning_log.append((column, method, value))

    def get_cleaned_df(self):
        return self.df

    def get_cleaning_log(self):
        return self.cleaning_log