"""
Entrypoint for the TablutClient module.
"""

import os
import argparse
from shared import strp_color, INITIAL_STATE, logger
from connectors.client import Client
from model.player import DQNPlayer

if __name__ == '__main__':
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Run the Tablut client.")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging.")
        args = parser.parse_args()

        # disable debug loggers if --debug wasn't specified
        disable_env_logger = False
        if not args.debug:
                disable_env_logger = True

        PLAYER_COLOR = os.environ['PLAYER_COLOR']
        TIMEOUT = os.environ['TIMEOUT']
        SERVER_IP = os.environ['SERVER_IP']
        WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']

        CHECKPOINTS_PATH = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "training",
                "checkpoints"
        )

        settings = {
                'current_state': INITIAL_STATE,
                'timeout': int(TIMEOUT),
                'server_ip': SERVER_IP,
                'port': int(WEBSOCKET_PORT)
        }

        player = DQNPlayer(color=strp_color(PLAYER_COLOR), disable_env_logger=disable_env_logger, from_pretrained=CHECKPOINTS_PATH)
        client = Client(player=player, settings=settings)
        client.main()
    except Exception as e:
        logger.error("An error occurred: %s", e)
