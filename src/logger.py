import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Clear existing handlers to avoid duplicate logs
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode='w') # Use 'w' to clear log on start
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Setup loggers
main_logger = setup_logger('main', 'main.log')
realtime_logger = setup_logger('realtime', 'realtime.log')
analysis_logger = setup_logger('analysis', 'analysis.log')
