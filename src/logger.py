import logging
import sys

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Clear existing handlers to avoid duplicate logs
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w') # Use 'w' to clear log on start
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Setup loggers
main_logger = setup_logger('main', 'main.log')
realtime_logger = setup_logger('realtime', 'realtime.log')
analysis_logger = setup_logger('analysis', 'analysis.log')
