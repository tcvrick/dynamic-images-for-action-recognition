import logging
import sys
from pathlib import Path


def initialize_logger(logger_name, log_dir):
    """
    Helper function for initializing a logger which writes to both file and stdout.
    """

    # Logging
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    log_filepath = log_dir / logger_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Log to file
    fh = logging.FileHandler('{}.log'.format(str(log_filepath)))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger
