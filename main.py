"""
    Entrypoint for the TablutClient.
"""

import os
from shared.utils.game_utils import *
from shared.huristic import *
from shared.utils import strp_state
"""
PLAYER_COLOR = os.environ['PLAYER_COLOR']
TIMEOUT = os.environ['TIMEOUT']
SERVER_IP = os.environ['SERVER_IP']
WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']
"""

stato = (
    "OOOBBBOOO\n"
    "OOOOBOOOO\n"
    "OOOOWOKOO\n"
    "BOOOWOOOB\n"
    "BBWWOWWBB\n"
    "BOOOWOOOB\n"
    "OOOOOOOOO\n"
    "OOOOOOOOO\n"
    "OOOBBBOOO\n"
)


move = Action(from_="c5", to_="c6", turn=Color.WHITE)

print(heuristic(INITIAL_STATE, move))