import logging
import logging.config
from .consts import LOGGING_CONFIG


# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create a logger using the configured dictionary
logger = logging.getLogger('tablut_logger')
