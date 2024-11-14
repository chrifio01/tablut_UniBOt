"""
This module defines constants used in the Tablut game setup.

Attributes:
    INITIAL_STATE (str): A string representation of the initial game state,
        with pieces arranged on the board.
    CAMPS (set of tuple): The set of board positions designated as 'camps' in Tablut.
        These positions are considered special areas with restricted movement rules.
"""

INITIAL_STATE = (
    "OOOBBBOOO\n"
    "OOOOBOOOO\n"
    "OOOOWOOOO\n"
    "BOOOWOOOB\n"
    "BBWWKWWBB\n"
    "BOOOWOOOB\n"
    "OOOOWOOOO\n"
    "OOOOBOOOO\n"
    "OOOBBBOOO\n"
    "-\n"
    "W"
)
"""
str: The initial configuration of the game board.
Rows of the board are separated by newline characters, with the
final line indicating the turn ('W' for white, 'B' for black).
Each character represents a piece or an empty space:
    - 'O' for an empty space.
    - 'B' for an attacker.
    - 'W' for a defender.
    - 'K' for the king.
    - 'T' for the throne (if used).
"""

CAMPS = {
    (0, 3), (0, 4), (0, 5), (1, 4),  # down
    (4, 1), (3, 0), (4, 0), (5, 0), # left
    (8, 3), (8, 4), (8, 5), (7, 4), # up
    (3, 8), (4, 8), (5, 8), (4, 7),  # right
}
"""
set of tuple: Positions designated as 'camps' on the board.
    These positions have specific movement restrictions
and are represented as (row, column) pairs on a 9x9 grid.
"""

# Define the configuration dictionary
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console'],
    },
    'loggers': {
        'my_debug_logger': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
    },
}
