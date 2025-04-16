import pandas as pd
import json
import io
import logging
from typing import Tuple, Optional, Any, Dict
import joblib
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

def export_dataset(df: pd.DataFrame, format: str) -> Tuple[bytes, str, str]:
    """
    Export dataset in various formats
    
    Args:
        df: DataFrame to export
        format: Export format ('csv', 'excel', 'json')
        
    Returns:
        Tuple containing:
        - Bytes of exported data
        - Filename
        - MIME type
        
    Raises:
        ValueError: If format is not supported
        RuntimeError: If export fails
    """
    try:
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Cannot export empty DataFrame")
            
        format = format.lower()
        if format not in ["csv", "excel", "json"]:
            raise ValueError(f"Unsupported format: {format}")

        # Export based on format
        if format == "csv":
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            return buffer.getvalue().encode('utf-8'), "processed_data.csv", "text/csv"
        
        elif format == "excel":
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            return buffer.getvalue(), "processed_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        elif format == "json":
            return df.to_json(orient="records").encode('utf-8'), "processed_data.json", "application/json"
            
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise RuntimeError(f"Failed to export dataset: {str(e)}")
    

def export_ml_artifacts(model: Any, filename: str) -> bytes:
    """Export ML model and related artifacts"""
    try:
        # Save model to bytes buffer
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        return buffer.getvalue(), f"{filename}.joblib", "application/octet-stream"
    except Exception as e:
        logger.error(f"Model export failed: {str(e)}")
        raise RuntimeError(f"Failed to export model: {str(e)}")


def export_quality_report(quality_metrics: dict, format: str = "markdown"):
    """
    Export data quality report in various formats

    Args:
        quality_metrics: Dictionary containing quality metrics
        format: Export format ('markdown', 'json', 'yaml')

    Returns:
        Tuple containing:
        - Bytes of exported report
        - Filename
        - MIME type
    """
    try:
        from datetime import datetime
        import json
        import yaml

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "markdown":
            nulls = quality_metrics.get('null_summary', {})
            null_lines = "\n".join(
                f"  - {col}: {info['null_count']} ({info['null_percentage']}%)"
                for col, info in nulls.items()
            )
            # Uniqueness section
            uniqueness = quality_metrics.get('uniqueness', {})
            uniq_lines = "\n".join(
                f"  - {col}: {info['unique_count']} ({info['unique_percentage']}%)"
                for col, info in uniqueness.items()
            )
            # Type consistency section
            mixed_types = quality_metrics.get('mixed_types', {})
            type_lines = "\n".join(
                f"  - {col}: {types}" for col, types in mixed_types.items()
            ) if mixed_types else "  - All columns consistent"
            # Value ranges section
            value_ranges = quality_metrics.get('value_ranges', {})
            range_lines = "\n".join(
                f"  - {col}: min={info['min']}, max={info['max']}"
                for col, info in value_ranges.items()
            )
            report = f"""# Data Quality Report

## Completeness
- Overall completeness: {quality_metrics.get('overall_completeness', 'N/A')}%
- Missing values by column:
{null_lines}

## Uniqueness
{uniq_lines}

## Data Type Consistency
{type_lines}

## Value Ranges
{range_lines}

Report generated: {datetime.now().isoformat()}
""".strip()
            return (
                report.encode('utf-8'),
                f"quality_report_{timestamp}.md",
                "text/markdown"
            )

        elif format == "json":
            return (
                json.dumps(quality_metrics, indent=2).encode('utf-8'),
                f"quality_report_{timestamp}.json",
                "application/json"
            )

        elif format == "yaml":
            return (
                yaml.dump(quality_metrics).encode('utf-8'),
                f"quality_report_{timestamp}.yaml",
                "text/yaml"
            )

        else:
            raise ValueError(f"Unsupported format: {format}")

    except Exception as e:
        logger.error(f"Quality report export failed: {str(e)}")
        raise RuntimeError(f"Failed to export quality report: {str(e)}")
def _format_dict(d: Dict) -> str:
    """Helper function to format dictionary for markdown"""
    return "\n".join([f"- {k}: {v}" for k, v in d.items()])

def _format_ranges(ranges: Dict) -> str:
    """Helper function to format range data for markdown"""
    lines = []
    for col, data in ranges.items():
        lines.append(f"### {col}")
        lines.append(f"- Range: {data['min']} to {data['max']}")
        lines.append(f"- Outliers: {data['outliers']['count']} ({data['outliers']['percentage']}%)")
    return "\n".join(lines)


def export_grouped_analysis(
    grouped_df: pd.DataFrame,
    group_cols: list,
    agg_funcs: list,
    format: str = "csv"
) -> Tuple[bytes, str, str]:
    """Export grouped analysis results"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            data = grouped_df.to_csv().encode('utf-8')
            filename = f"grouped_analysis_{timestamp}.csv"
            mimetype = "text/csv"
        elif format == "excel":
            buffer = io.BytesIO()
            grouped_df.to_excel(buffer)
            data = buffer.getvalue()
            filename = f"grouped_analysis_{timestamp}.xlsx"
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif format == "json":
            data = grouped_df.to_json(orient="split").encode('utf-8')
            filename = f"grouped_analysis_{timestamp}.json"
            mimetype = "application/json"
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return data, filename, mimetype
        
    except Exception as e:
        logger.error(f"Failed to export grouped analysis: {str(e)}")
        raise RuntimeError(f"Export failed: {str(e)}")