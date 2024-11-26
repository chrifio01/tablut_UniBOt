from shared import strp_state, INITIAL_STATE

from shared import Color, State
from model.player import DQNPlayer

# Create an instance of DQNPlayer
player = DQNPlayer(color=Color.WHITE)

# Predict the best action for the given state
predicted_action = player.fit(strp_state(INITIAL_STATE))

# Print the predicted action
print(f"Predicted action: {predicted_action}")

"""
Entrypoint for the TablutClient module.
"""
"""

import os
from shared.utils import strp_color
from shared.consts import INITIAL_STATE
from shared.loggers import logger
from connectors.client import Client
from model.player import DQNPlayer

if __name__ == '__main__':
    try:
        PLAYER_COLOR = os.environ['PLAYER_COLOR']
        TIMEOUT = os.environ['TIMEOUT']
        SERVER_IP = os.environ['SERVER_IP']
        WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']

        settings = {
            'current_state': INITIAL_STATE,
            'timeout': int(TIMEOUT),
            'server_ip': SERVER_IP,
            'port': int(WEBSOCKET_PORT)
        }
        player = DQNPlayer(color=strp_color(PLAYER_COLOR))
        client = Client(player=player, settings=settings)
        client.main()
    except Exception as e:
        logger.error("An error occurred: %s", e)
"""
