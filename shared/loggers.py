"""
This module sets up a logger for the application using a predefined configuration dictionary.

The logger is configured to output logs to both the console and a file, depending on the settings 
defined in the `LOGGING_CONFIG` dictionary imported from the `consts` module. 

Attributes:
    logger (logging.Logger): The primary logger instance configured for this application, 
                             accessible as `tablut_logger`.

Usage:
    Import this module to use the pre-configured `logger` in other parts of the application. 
    For example:
        from your_module.logging_setup import logger
        logger.debug("This is a debug message")
    
    The `LOGGING_CONFIG` should define the handlers, formatters, and log levels for this logger.
"""

import logging.config
from .consts import LOGGING_CONFIG

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create a logger using the configured dictionary
logger = logging.getLogger('tablut_logger')
training_logger = logging.getLogger('training_logger')
env_logger = logging.getLogger('env_logger')
