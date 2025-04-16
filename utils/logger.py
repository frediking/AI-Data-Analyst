import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logging():
    """Configure application logging"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging with rotation (5 MB per file, keep 3 backup files)
    handlers = [
        RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler()  # Also print to console
    ]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    
    # Create logger
    logger = logging.getLogger('FrozAI')
    return logger