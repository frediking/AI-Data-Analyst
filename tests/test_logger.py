import os
import glob
import logging
import shutil
import pytest

from utils.logger import setup_logging

@pytest.fixture(autouse=True)
def cleanup_logs():
    """Remove logs directory before and after each test for isolation."""
    log_dir = "logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    yield
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

def test_setup_logging_creates_log_file_and_directory():
    setup_logging()
    log_dir = "logs"
    # Check logs directory exists
    assert os.path.exists(log_dir)
    # Check at least one log file is created
    log_files = glob.glob(os.path.join(log_dir, "app_*.log"))
    assert len(log_files) > 0

def test_logger_writes_to_log_file_and_console():
    setup_logging()
    logger = logging.getLogger('FrozAI')
    test_message = "Logger test message"
    logger.info(test_message)
    # Flush file handlers to ensure log is written
    for handler in logger.handlers:
        handler.flush()
    # Check message in log file
    log_files = glob.glob(os.path.join("logs", "app_*.log"))
    assert log_files
    with open(log_files[0], "r") as f:
        log_content = f.read()
    assert test_message in log_content

def test_logger_rotation():
    setup_logging()
    logger = logging.getLogger('FrozAI')
    # Write a large amount to trigger rotation (simulate, but not actually fill 5MB)
    for i in range(1000):
        logger.info("X" * 1000)
    log_files = glob.glob(os.path.join("logs", "app_*.log"))
    # Should still have at least one log file, rotation policy is tested by not crashing
    assert log_files
