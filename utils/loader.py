from typing import List, Optional
from pathlib import Path
from langchain.document_loaders import CSVLoader
from langchain.schema import Document
import pandas as pd

def load_csv_documents(
    file_path: str,
    encoding: str = 'utf-8',
    csv_args: Optional[dict] = None
) -> List[Document]:
    """
    Loads a CSV file into LangChain Documents with enhanced error handling and options.
    
    Args:
        file_path (str): Path to the CSV file
        encoding (str): File encoding (default: utf-8)
        csv_args (dict, optional): Additional arguments for pandas read_csv
        
    Returns:
        List[Document]: List of LangChain Document objects
        
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
        
        # Initialize loader with custom configuration
        loader = CSVLoader(
            str(file_path),
            encoding=encoding,
            csv_args=csv_args
        )
        
        # Load documents
        documents = loader.load()
        
        # Validate output
        if not documents:
            raise ValueError("No data was loaded from the CSV file")
            
        return documents
        
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")
    except UnicodeDecodeError:
        raise ValueError(f"Failed to decode file with {encoding} encoding. Try a different encoding.")
    except Exception as e:
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
        raise Exception(f"Error previewing CSV file: {str(e)}")