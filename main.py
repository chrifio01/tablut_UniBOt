"""
    Entrypoint for the TablutClient.
"""
from shared.random_player import RandomPlayer
from shared.utils import Color
from dotenv import load_dotenv
import os

load_dotenv()


"""
# PLAYER_COLOR = os.environ['PLAYER_COLOR']
# TIMEOUT = os.environ['TIMEOUT']
SERVER_IP = os.environ['SERVER_IP']
# WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']

from connectors.client import Client

white_player = RandomPlayer(color=Color.WHITE)
black_player = RandomPlayer(color=Color.BLACK)

white_client = Client(player=white_player, server_ip=SERVER_IP, port=5800)
black_client = Client(player=black_player, server_ip=SERVER_IP, port=5801)

white_client.send_name()
black_client.send_name()

white_client.read_state()
white_client.send_move({"from": "e4", "to": "e5"})

black_client.read_state()
black_client.send_move({"from": "e7", "to": "e3"})
"""