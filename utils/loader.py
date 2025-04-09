from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_csv_documents(
    file_path: str,
    encoding: str = 'utf-8',
    csv_args: Optional[dict] = None
) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame with enhanced error handling and options.
    
    Args:
        file_path (str): Path to the CSV file
        encoding (str): File encoding (default: utf-8)
        csv_args (dict, optional): Additional arguments for pandas read_csv
        
    Returns:
        pd.DataFrame: Loaded CSV data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
        Exception: For other processing errors
    """
    try:
        # Validate file path
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        # Set default CSV arguments
        csv_args = csv_args or {
            'encoding': encoding,
            'on_bad_lines': 'warn',
            'low_memory': False
        }
        
        # Load DataFrame
        df = pd.read_csv(str(file_path), **csv_args)
        
        # Validate output
        if df.empty:
            raise ValueError("No data was loaded from the CSV file")
            
        return df
        
    except pd.errors.EmptyDataError:
        logger.error("Empty CSV file encountered")
        raise ValueError("The CSV file is empty")
    except UnicodeDecodeError:
        logger.error(f"Encoding error with {encoding}")
        raise ValueError(f"Failed to decode file with {encoding} encoding. Try a different encoding.")
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise Exception(f"Error loading CSV file: {str(e)}")

def get_csv_preview(file_path: str, nrows: int = 5) -> pd.DataFrame:
    """
    Get a preview of the CSV file without loading the entire dataset.
    
    Args:
        file_path (str): Path to the CSV file
        nrows (int): Number of rows to preview
        
    Returns:
        pd.DataFrame: Preview of the CSV data
    """
    try:
        return pd.read_csv(file_path, nrows=nrows)
    except Exception as e:
        logger.error(f"Error previewing CSV: {str(e)}")
        raise Exception(f"Error previewing CSV file: {str(e)}")

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate loaded DataFrame and return basic statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict[str, Any]: Basic statistics and validation results
    """
    try:
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
        return stats
    except Exception as e:
        logger.error(f"Error validating DataFrame: {str(e)}")
        raise ValueError(f"Failed to validate DataFrame: {str(e)}")