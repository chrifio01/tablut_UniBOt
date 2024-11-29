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
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'app.log',
        },
        'training_file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'training.log',
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console'],
    },
    'loggers': {
        'tablut_logger': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
        'training_logger': {
            'level': 'DEBUG',
            'handlers': ['training_file', 'console'],
            'propagate': False,
        },
        'env_logger': {
            'level': 'DEBUG',
            'handlers': ['training_file', 'console'],
            'propagate': False,
        },
    },
}

"""
Weights for the heuristic function
"""

WEIGHTS = [[0, 20, 20, -6, -6, -6, 20, 20, 0],
           [20, 1, 1, -5, -6, -5, 1,  1, 20],
           [20, 1, 4,  1, -2,  1, 4,  1, 20],
           [-6, -5, 1,  1,  1,  1, 1, -5, -6],
           [-6, -6, -2,  1,  2,  1, -2, -6, -6],
           [-6, -5, 1,  1,  1,  1, 1, -5, -6],
           [20, 1, 4,  1, -2,  1, 4,  1, 20],
           [20, 1, 1, -5, -6, -5, 1,  1, 20],
           [0, 20, 20, -6, -6, -6, 20, 20, 0]]

ALPHA_W, BETA_W, GAMMA_W, THETA_W, EPSILON_W, OMEGA_W = [0.21639120828483156, 0.723587137336777, 9, 1.06923818569000507, 2.115749207248323, 10]

ALPHA_B, BETA_B, GAMMA_B, THETA_B, EPSILON_B = [0.958245251997756, 0.25688393654958275, 0.812052344592159, 0.9193347856045799, 1.7870310915100207]

WIN_TILES = [(0,1),(0,2),(0,6),(0,7),(1,0),(2,0),(6,0),(7,0),(8,1),(8,2),(8,6),(8,7),(1,8),(2,8),(6,8),(7,8)]

# These lines are defining constants for rewards and punishments in the Tablut game. Here's what each
# constant represents:
WIN_REWARD = 20

LOSS_REWARD = - 20

DRAW_REWARD = - 1

INVALID_ACTION_PUNISHMENT = - 20

DEFENDER_NUM = 8

ATTACKER_NUM = 16
