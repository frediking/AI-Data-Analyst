import pandas as pd
import numpy as np

def detect_outliers(series: pd.Series) -> dict:
    """Detect outliers using the IQR method."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return {'count': 0, 'percentage': 0.0}
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        'count': int(outliers.count()),
        'percentage': round((outliers.count() / len(series) * 100), 2)
    }

def assess_data_quality(df):
    """
    Assess data quality metrics for a pandas DataFrame.
    Returns:
        dict: Quality metrics.
    Raises:
        TypeError: If input is not a DataFrame.
        ValueError: If DataFrame is empty or contains only null values.
    """
    # Type check
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Empty DataFrame check
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # All-null DataFrame check
    if df.isnull().all().all():
        raise ValueError("Input DataFrame contains only null values")

    metrics = {}

    # 1. Completeness
    total_cells = df.shape[0] * df.shape[1]
    completeness = round((1 - df.isnull().sum().sum() / total_cells) * 100, 2) if total_cells else 0.0
    metrics['overall_completeness'] = completeness

    # 2. Type Consistency (mixed types)
    type_consistency = {}
    for col in df.columns:
        types = set(type(x) for x in df[col] if pd.notnull(x))
        if len(types) > 1:
            type_consistency[col] = [str(t) for t in types]
    metrics['type_consistency'] = type_consistency

    # 3. Uniqueness (per column)
    uniqueness = {}
    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        total_count = df[col].dropna().shape[0]
        uniqueness[col] = {
            'unique_count': int(unique_count),
            'unique_percentage': round((unique_count / total_count * 100), 2) if total_count else 0.0
        }
    metrics['uniqueness'] = uniqueness

    # 4. Outlier Detection (numeric columns)
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        outliers[col] = detect_outliers(df[col])
    metrics['outliers'] = outliers

    # 5. Value Ranges (numeric columns)
    value_ranges = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if not series.empty:
            value_ranges[col] = {
                'min': float(series.min()),
                'max': float(series.max())
            }
        else:
            value_ranges[col] = {'min': None, 'max': None}
    metrics['value_ranges'] = value_ranges

    # 6. Null/NA summary (per column)
    null_summary = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_summary[col] = {
            'null_count': int(null_count),
            'null_percentage': round((null_count / len(df) * 100), 2) if len(df) else 0.0
        }
    metrics['null_summary'] = null_summary

    return metrics


def generate_quality_report(df: pd.DataFrame) -> str:
    """Generate a formatted quality report from assess_data_quality output."""
    metrics = assess_data_quality(df)

    report = "# Data Quality Assessment Report\n\n"

    # Completeness Section
    report += "## 1. Completeness\n"
    report += f"- Overall completeness: {metrics['overall_completeness']}%\n"
    report += "- Columns with missing values:\n"
    for col, null_info in metrics['null_summary'].items():
        if null_info['null_count'] > 0:
            report += f"  * {col}: {null_info['null_percentage']}% missing ({null_info['null_count']} rows)\n"

    # Uniqueness Section
    report += "\n## 2. Uniqueness\n"
    for col, uniq in metrics['uniqueness'].items():
        report += f"  * {col}: {uniq['unique_percentage']}% unique ({uniq['unique_count']} unique values)\n"

    # Type Consistency Section
    report += "\n## 3. Data Type Consistency\n"
    for col, types in metrics['type_consistency'].items():
        report += f"- {col}: {', '.join(types)}\n"

    # Value Ranges Section
    report += "\n## 4. Value Ranges (Numeric Columns)\n"
    for col, range_data in metrics['value_ranges'].items():
        report += f"\n### {col}\n"
        report += f"- Range: {range_data['min']} to {range_data['max']}\n"
        if col in metrics['outliers']:
            out = metrics['outliers'][col]
            report += f"- Outliers: {out['count']} ({out['percentage']}%)\n"

    return report

def assess_grouped_quality(df: pd.DataFrame, group_cols: list, numeric_cols: list) -> dict:
    """Assess data quality for grouped data with robust column checks."""
    # Validate column existence
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"Group column '{col}' not found in DataFrame")
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Numeric column '{col}' not found in DataFrame")

    try:
        quality_metrics = {
            'group_completeness': {},
            'group_statistics': {},
            'outliers_by_group': {}
        }
        
        grouped = df.groupby(group_cols)
        
        # Assess completeness by group
        quality_metrics['group_completeness'] = grouped[numeric_cols].count().to_dict()
        
        # Basic statistics by group
        quality_metrics['group_statistics'] = grouped[numeric_cols].describe().to_dict()
        
        # Detect outliers within groups
        for name, group in grouped:
            quality_metrics['outliers_by_group'][name] = {
                col: detect_outliers(group[col]) for col in numeric_cols
            }
            
        return quality_metrics
        
    except Exception as e:
        raise RuntimeError(f"Failed to assess grouped data quality: {str(e)}")
    

    # ...existing code...

def analyze_group_quality(df: pd.DataFrame, group_cols: list, agg_cols: list) -> dict:
    """
    Analyze data quality metrics for grouped data with robust column checks.
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        agg_cols: Columns to analyze
    Returns:
        Dictionary containing group-level quality metrics
    """
    # Validate column existence
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"Group column '{col}' not found in DataFrame")
    for col in agg_cols:
        if col not in df.columns:
            raise ValueError(f"Aggregate column '{col}' not found in DataFrame")

    try:
        metrics = {}
        grouped = df.groupby(group_cols)
        
        # Completeness by group
        metrics['completeness'] = {
            'missing_by_group': grouped[agg_cols].isnull().sum().to_dict(),
            'complete_by_group': grouped[agg_cols].notnull().sum().to_dict()
        }
        
        # Basic statistics by group
        metrics['statistics'] = grouped[agg_cols].agg(['mean', 'std', 'min', 'max']).to_dict()
        
        # Group sizes
        metrics['group_sizes'] = grouped.size().to_dict()
        
        return metrics
        
    except Exception as e:
        raise RuntimeError(f"Group analysis failed: {str(e)}")

