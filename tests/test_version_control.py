import pytest
import shutil
import os
import pandas as pd
from utils.version_control import DataVersionControl

def test_version_control_init():
    """Test basic initialization of version control"""
    vc = DataVersionControl(base_dir="test_versions")
    assert vc is not None
    assert vc.base_dir == "test_versions"

def test_version_save_load():
    """Test saving and loading a version"""
    # Create test DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    
    # Initialize version control
    vc = DataVersionControl(base_dir="test_versions")
    
    # Save version
    version_id, version_hash = vc.save_version(
        df=df,
        description="Test version",
        transformation_notes=["Initial test data"]
    )
    
    # Load version
    loaded_df = vc.load_version(version_id)
    pd.testing.assert_frame_equal(loaded_df, df)
    
    # Cleanup
    shutil.rmtree("test_versions")