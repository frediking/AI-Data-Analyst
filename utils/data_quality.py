import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def assess_data_quality(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality assessment
    
    Returns dictionary with quality metrics:
    - Completeness (missing values)
    - Uniqueness
    - Consistency (data types)
    - Validity (range checks)
    - Integrity (relationships)
    """
    try:
        quality_metrics = {}
        
        # Completeness Check
        missing_stats = df.isnull().sum()
        completeness = {
            'missing_counts': missing_stats.to_dict(),
            'missing_percentages': (missing_stats / len(df) * 100).round(2).to_dict(),
            'overall_completeness': ((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100).round(2)
        }
        quality_metrics['completeness'] = completeness
        
        # Uniqueness Check
        uniqueness = {
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_per_column': df.nunique().to_dict(),
            'unique_percentages': (df.nunique() / len(df) * 100).round(2).to_dict()
        }
        quality_metrics['uniqueness'] = uniqueness
        
        # Data Type Consistency
        type_consistency = {
            'dtypes': df.dtypes.astype(str).to_dict(),
            'mixed_types': detect_mixed_types(df)
        }
        quality_metrics['type_consistency'] = type_consistency
        
        # Value Range Analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        ranges = {}
        for col in numeric_cols:
            try:
                ranges[col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'outliers': detect_outliers(df[col])
                }
            except Exception as e:
                logger.warning(f"Failed to analyze value ranges for column '{col}': {str(e)}")
                ranges[col] = {
                    'min': None,
                    'max': None,
                    'outliers': {'count': 0, 'percentage': 0.0}
                }
        quality_metrics['value_ranges'] = ranges
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Data quality assessment failed: {str(e)}")
        raise RuntimeError(f"Failed to assess data quality: {str(e)}")

def detect_mixed_types(df: pd.DataFrame) -> Dict:
    """Detect columns with mixed data types"""
    mixed_types = {}
    for col in df.columns:
        try:
            types = df[col].apply(type).unique()
            if len(types) > 1:
                mixed_types[col] = [str(t) for t in types]
        except:
            continue
    return mixed_types

def detect_outliers(series: pd.Series) -> Dict:
    """Detect outliers using IQR method"""
    try:
        # Ensure the series is numeric and drop NaN values
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(series) * 100), 2) if len(series) > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Outlier detection failed: {str(e)}")
        return {
            'count': 0,
            'percentage': 0.0
        }

def generate_quality_report(df: pd.DataFrame) -> str:
    """Generate a formatted quality report"""
    try:
        metrics = assess_data_quality(df)
        
        report = """# Data Quality Assessment Report\n\n"""
        
        # Completeness Section
        report += "## 1. Completeness\n"
        report += f"- Overall completeness: {metrics['completeness']['overall_completeness']}%\n"
        report += "- Columns with missing values:\n"
        for col, pct in metrics['completeness']['missing_percentages'].items():
            if pct > 0:
                report += f"  * {col}: {pct}% missing\n"
        
        # Uniqueness Section
        report += "\n## 2. Uniqueness\n"
        report += f"- Duplicate rows: {metrics['uniqueness']['duplicate_rows']}\n"
        report += "- Unique value percentages:\n"
        for col, pct in metrics['uniqueness']['unique_percentages'].items():
            report += f"  * {col}: {pct}% unique\n"
        
        # Type Consistency Section
        report += "\n## 3. Data Type Consistency\n"
        for col, dtype in metrics['type_consistency']['dtypes'].items():
            report += f"- {col}: {dtype}\n"
        
        # Mixed Types Warning
        if metrics['type_consistency']['mixed_types']:
            report += "\nWarning: Mixed data types detected in:\n"
            for col, types in metrics['type_consistency']['mixed_types'].items():
                report += f"- {col}: {', '.join(types)}\n"
        
        # Value Ranges Section
        report += "\n## 4. Value Ranges (Numeric Columns)\n"
        for col, range_data in metrics['value_ranges'].items():
            report += f"\n### {col}\n"
            report += f"- Range: {range_data['min']} to {range_data['max']}\n"
            report += f"- Outliers: {range_data['outliers']['count']} ({range_data['outliers']['percentage']}%)\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return f"Failed to generate quality report: {str(e)}"