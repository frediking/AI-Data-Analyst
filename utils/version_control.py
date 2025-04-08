import os
import pandas as pd
from datetime import datetime
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
import streamlit as st

logger = logging.getLogger(__name__)

class DataVersionControl:
    """Handles data versioning and transformation tracking"""
    
    def __init__(self, base_dir: str = "data_versions"):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "version_metadata.json")
        self.initialize_storage()

    def initialize_storage(self) -> None:
        """Initialize version storage structure"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            if not os.path.exists(self.metadata_file):
                self._save_metadata({})
                logger.info("Initialized version control storage")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}")
            raise RuntimeError(f"Storage initialization failed: {str(e)}")

    def _generate_version_hash(self, df: pd.DataFrame) -> str:
        """Generate unique hash for DataFrame version"""
        try:
            # Create deterministic string representation
            df_string = pd.util.hash_pandas_object(df).sum()
            return hashlib.sha256(str(df_string).encode()).hexdigest()[:12]
        except Exception as e:
            logger.error(f"Hash generation failed: {str(e)}")
            raise ValueError(f"Could not generate version hash: {str(e)}")

    def _save_metadata(self, metadata: Dict) -> None:
        """Save version metadata to JSON"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise IOError(f"Could not save metadata: {str(e)}")

    def _load_metadata(self) -> Dict:
        """Load version metadata from JSON"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return {}

    def save_version(self, 
                    df: pd.DataFrame, 
                    description: str, 
                    transformation_notes: List[str]) -> Tuple[str, str]:
        """
        Save a new version of the DataFrame
        
        Args:
            df: DataFrame to version
            description: Version description
            transformation_notes: List of transformations applied
            
        Returns:
            Tuple of (version_id, version_hash)
        """
        try:
            # Generate version information
            version_hash = self._generate_version_hash(df)
            timestamp = datetime.now().isoformat()
            version_id = f"v{timestamp.replace(':', '').replace('.', '')}"
            
            # Save DataFrame
            version_path = os.path.join(self.base_dir, f"{version_id}.parquet")
            df.to_parquet(version_path)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata[version_id] = {
                "hash": version_hash,
                "timestamp": timestamp,
                "description": description,
                "transformations": transformation_notes,
                "shape": df.shape,
                "columns": list(df.columns),
                "file_path": version_path
            }
            self._save_metadata(metadata)
            
            logger.info(f"Saved version {version_id} with hash {version_hash}")
            return version_id, version_hash
            
        except Exception as e:
            logger.error(f"Version saving failed: {str(e)}")
            raise RuntimeError(f"Failed to save version: {str(e)}")

    def load_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """Load specific version of DataFrame"""
        try:
            metadata = self._load_metadata()
            if version_id not in metadata:
                raise ValueError(f"Version {version_id} not found")
                
            version_path = metadata[version_id]["file_path"]
            df = pd.read_parquet(version_path)
            logger.info(f"Loaded version {version_id}")
            return df
            
        except Exception as e:
            logger.error(f"Version loading failed: {str(e)}")
            raise RuntimeError(f"Failed to load version: {str(e)}")

    def get_version_history(self) -> List[Dict]:
        """Get sorted list of all versions with metadata"""
        try:
            metadata = self._load_metadata()
            versions = [
                {"id": k, **v} 
                for k, v in metadata.items()
            ]
            return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            logger.error(f"Failed to get version history: {str(e)}")
            return []
        