import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report
from typing import Dict, Tuple, List, Any, Union
import logging

logger = logging.getLogger(__name__)

class MLInsights:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
    def identify_task_type(self, target_col: str) -> str:
        """Determine if this is a classification or regression task"""
        unique_values = self.df[target_col].nunique()
        if unique_values <= 10:  # Arbitrary threshold
            return 'classification'
        return 'regression'
    
    def prepare_data(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling"""
        try:
            # Remove rows with missing values
            df_clean = self.df.dropna()
            
            # Encode categorical variables
            X = pd.get_dummies(df_clean.drop(columns=[target_col]))
            y = df_clean[target_col]
            
            return X, y
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def quick_model(self, target_col: str) -> Dict[str, Any]:
        """Generate quick ML insights"""
        try:
            task_type = self.identify_task_type(target_col)
            X, y = self.prepare_data(target_col)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                performance = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                performance = {
                    'r2_score': r2_score(y_test, y_pred)
                }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'task_type': task_type,
                'performance': performance,
                'feature_importance': feature_importance,
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Quick model generation failed: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame) -> Union[go.Figure, None]:
        """Plot feature importance"""
        try:
            fig = px.bar(
                feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            fig.update_layout(showlegend=False)
            return fig
        except Exception as e:
            logger.error(f"Feature importance plotting failed: {str(e)}")
            return None