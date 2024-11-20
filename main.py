"""
    Entrypoint for the TablutClient.
"""
import os
from shared.random_player import RandomPlayer
from shared.utils import strp_color, parse_yaml
from shared.consts import INITIAL_STATE
from shared.loggers import logger
from connectors.client import Client

_config_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "config.yaml"
)

CONFIG = parse_yaml(_config_file_path)  # env and model configs

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
        player = RandomPlayer(color=strp_color(PLAYER_COLOR))
        client = Client(player=player, settings=settings)
    
        client.main()
    except Exception as e:
        logger.error("An error occurred: %s", e)
        