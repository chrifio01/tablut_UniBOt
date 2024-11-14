"""
    Entrypoint for the TablutClient.
"""

import os
from shared.utils.game_utils import *
from shared.heuristic import *
from shared.utils import strp_state
"""
PLAYER_COLOR = os.environ['PLAYER_COLOR']
TIMEOUT = os.environ['TIMEOUT']
SERVER_IP = os.environ['SERVER_IP']
WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']
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

state = strp_state(INITIAL_STATE)
move = Action(from_="c5", to_="c6", turn=Color.WHITE)



print(heuristic(state, move))