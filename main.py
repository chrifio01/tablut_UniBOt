"""
    Entrypoint for the TablutClient.
"""
import os
from shared.random_player import RandomPlayer
from shared.utils import strp_color
from shared.consts import INITIAL_STATE
from connectors.client import Client

PLAYER_COLOR = os.environ['PLAYER_COLOR']
TIMEOUT = os.environ['TIMEOUT']
SERVER_IP = os.environ['SERVER_IP']
WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']

if __name__ == '__main__':
    player = RandomPlayer(color=strp_color(PLAYER_COLOR))
    client = Client(player=player, server_ip=SERVER_IP, port=int(WEBSOCKET_PORT), current_state=INITIAL_STATE, timeout=int(TIMEOUT))
    client.main()