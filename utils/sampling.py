import pandas as pd
import numpy as np
from typing import Union, List, Dict
from sklearn.model_selection import train_test_split


def stratified_sample(
    df: pd.DataFrame, 
    stratify_col: str,
    sample_size: Union[int, float],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Stratified sampling that maintains class distribution
    """
    if isinstance(sample_size, float):
        if not 0 < sample_size <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        return df.groupby(stratify_col, group_keys=False)\
                 .apply(lambda x: x.sample(frac=sample_size, random_state=random_state))
    else:
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("Sample size must be a positive integer")
        return df.groupby(stratify_col, group_keys=False)\
                 .apply(lambda x: x.sample(n=min(sample_size, len(x)), random_state=random_state))


def time_based_sample(
    df: pd.DataFrame,
    time_col: str,
    freq: str = 'D',
    sample_size: Union[int, float] = 0.1,
    method: str = 'systematic'
) -> pd.DataFrame:
    """
    Time-based sampling methods
    
    Args:
        df: Input DataFrame (must have datetime column)
        time_col: Name of datetime column
        freq: Sampling frequency (e.g. 'D'=daily, 'H'=hourly)
        sample_size: Either:
            - Integer: exact number of samples
            - Float: fraction of data (0.0-1.0)
        method: Sampling method:
            - 'systematic': Fixed interval sampling
            - 'random': Random time points
            
    Returns:
        Sampled DataFrame
    """
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values(time_col)
    
    if method == 'systematic':
        # Calculate sampling interval
        total = len(df)
        if isinstance(sample_size, float):
            interval = int(1 / sample_size)
        else:
            interval = int(total / sample_size)
        
        return df.iloc[::interval].copy()
    
    elif method == 'random':
        if isinstance(sample_size, float):
            return df.sample(frac=sample_size)
        return df.sample(n=sample_size)
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def get_sampling_methods() -> Dict[str, str]:
    """Returns available sampling methods with descriptions"""
    return {
        'stratified': 'Maintains class distribution from specified column',
        'time_based': 'Samples based on time intervals (systematic/random)',
        'random': 'Simple random sampling',
        'cluster': 'Cluster sampling (not yet implemented)'
    }
