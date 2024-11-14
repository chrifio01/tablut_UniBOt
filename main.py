"""
    Entrypoint for the TablutClient.
"""

import os
from shared.utils.game_utils import *
from shared.huristic import *
from shared.utils import strp_state

PLAYER_COLOR = os.environ['PLAYER_COLOR']
TIMEOUT = os.environ['TIMEOUT']
SERVER_IP = os.environ['SERVER_IP']
WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']
