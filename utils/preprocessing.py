import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import logging
from .data_quality import assess_data_quality

logger = logging.getLogger(__name__)

def clean_dataset(
    df: pd.DataFrame,
    reset_index: bool = True,
    standardize_dates: bool = True,
    clean_numeric: bool = True,
    clean_text: bool = True,
    remove_duplicates: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean and validate the input dataset

    Args:
        df: Input DataFrame
        clean_numeric: Whether to clean numeric columns
        clean_text: Whether to clean text columns
        remove_duplicates: Whether to remove duplicate rows

    Returns:
        Tuple containing:
        - Cleaned DataFrame
        - List of cleaning operations performed
    """
    try:
        cleaning_notes = []
        df_clean = df.copy()

        # Track initial state for version control
        initial_hash = pd.util.hash_pandas_object(df).sum()

        # Handle all-null DataFrame gracefully
        if df_clean.isnull().all().all():
            cleaning_notes.append("Input DataFrame contained only null values; returned empty DataFrame.")
            return pd.DataFrame(), cleaning_notes

        # Get initial quality metrics
        initial_metrics = assess_data_quality(df)
        initial_completeness = initial_metrics['overall_completeness']

        if reset_index:
            # Store original index type for logging
            df_clean = df_clean.reset_index(drop=True)
            cleaning_notes.append("Reset DataFrame index")

        if standardize_dates:
            # Try to standardize date columns
            date_cols = [col for col in df_clean.columns if 'date' in col.lower()]
            for col in date_cols:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    cleaning_notes.append(f"Standardized dates in column {col}")
                except Exception:
                    cleaning_notes.append(f"Could not standardize dates in column {col}")

        if clean_numeric:
            # Handle numeric columns
            numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
            missing_counts = df_clean[numeric_cols].isnull().sum()
            if missing_counts.any():
                cleaning_notes.append(f"Found {missing_counts.sum()} missing values in numeric columns")
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())

        if clean_text:
            # Handle text columns
            text_cols = df_clean.select_dtypes(include=['object']).columns
            for col in text_cols:
                try:
                    # Try converting to numeric if possible
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='raise')
                    cleaning_notes.append(f"Converted {col} to numeric")
                except ValueError:
                    # If not numeric, clean text (strip whitespace, replace empty/whitespace-only with NaN)
                    orig_non_null = df_clean[col].notnull().sum()
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    empty_mask = df_clean[col].isin(['', 'None', 'nan'])
                    if empty_mask.any():
                        df_clean.loc[empty_mask, col] = None
                        cleaning_notes.append(f"Replaced empty or whitespace-only values in column {col} with NaN")
                    cleaning_notes.append(f"Cleaned text in column {col}")

        if remove_duplicates:
            # Remove duplicate rows
            dups = df_clean.duplicated().sum()
            if dups > 0:
                df_clean = df_clean.drop_duplicates()
                cleaning_notes.append(f"Removed {dups} duplicate rows")

        # Remove constant columns (columns with only one unique value)
        constant_cols = [col for col in df_clean.columns if df_clean[col].nunique(dropna=True) == 1]
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            cleaning_notes.append(f"Removed constant columns: {constant_cols}")

        if not cleaning_notes:
            cleaning_notes.append("No cleaning operations were necessary")

        # Get final quality metrics
        final_metrics = assess_data_quality(df_clean)
        final_completeness = final_metrics['overall_completeness']

        if final_completeness < initial_completeness:
            logger.warning("Dataset completeness decreased after cleaning operations")
        else:
            logger.info("Dataset completeness improved after cleaning operations")

        # Track final state for version control
        final_hash = pd.util.hash_pandas_object(df_clean).sum()
        if initial_hash != final_hash:
            logger.info("Dataset hash changed after cleaning operations")
        else:
            logger.info("Dataset hash remains unchanged after cleaning operations")
        return df_clean, cleaning_notes

    except Exception as e:
        logger.error(f"Dataset cleaning failed: {str(e)}")
        raise RuntimeError(f"Failed to clean dataset: {str(e)}")
    

        