"""
Entrypoint for the TablutClient module.
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
        
        PRETRAINED_MODEL_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model",
            "dqn_agent.zip"
        )

        settings = {
            'current_state': INITIAL_STATE,
            'timeout': int(TIMEOUT),
            'server_ip': SERVER_IP,
            'port': int(WEBSOCKET_PORT)
        }
        player = DQNPlayer(color=strp_color(PLAYER_COLOR), disable_env_logger=True, from_pretrained=PRETRAINED_MODEL_PATH)
        client = Client(player=player, settings=settings)
        client.main()
    except Exception as e:
        logger.error("An error occurred: %s", e)
