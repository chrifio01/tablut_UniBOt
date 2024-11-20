"""
    Entrypoint for the TablutClient.
"""
import os
from shared.random_player import RandomPlayer
from shared.utils import strp_color
from shared.consts import INITIAL_STATE
from shared.loggers import logger
from connectors.client import Client

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



from environment.tablut import Environment
from shared.history import History, Match
from shared.utils import AbstractPlayer
from shared.utils.game_utils import Color, Action, Board
from shared.heuristic import heuristic
from shared.utils.env_utils import strp_state
STATE1 = (
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
STATE2 = (
    "OOOBBBOOO\n"
    "OOOOBOOOO\n"
    "OOOOWWOOO\n"
    "BOOOWOOOB\n"
    "BBWWKOWBB\n"
    "BOOOWOOOB\n"
    "OOOOWOOOO\n"
    "OOOOBOOOO\n"
    "OOOBBBOOO\n"
    "-\n"
    "B"
)
STATE3 = (
    "OOOBBBOOO\n"
    "OOOOBOOOO\n"
    "OOOOWWOOO\n"
    "BOOOWOOOB\n"
    "BBWWKOWBB\n"
    "BOOOWOOOB\n"
    "OOOOWOOOO\n"
    "OOOOOBOOO\n"
    "OOOBBBOOO\n"
    "-\n"
    "W"
)
STATE4 = (
    "OOOBBBOOO\n"
    "OOOOBOOOO\n"
    "OOOOWOOOO\n"
    "BOOOWOOOB\n"
    "BBWWKWWBB\n"
    "BOOOWOOOB\n"
    "OOOOWOOOO\n"
    "OOOOOBOOO\n"
    "OOOBBBOOO\n"
    "-\n"
    "B"
)
STATE5 = (
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
p1 = AbstractPlayer(STATE1, "testw", Color.WHITE)
p2 = AbstractPlayer(STATE1, "testb", Color.BLACK)
turns = [(STATE1, Action(from_= "f5", to_="f7", turn=Color.WHITE), heuristic(STATE1,  Action(from_= "f5", to_="f7", turn=Color.WHITE)))]
turns.append((STATE2, Action(from_= "e2", to_="f2", turn=Color.BLACK), heuristic(STATE1,  Action(from_= "e2", to_="f2", turn=Color.BLACK))))
turns.append((STATE3, Action(from_= "f7", to_="f5", turn=Color.WHITE), heuristic(STATE1,  Action(from_= "f7", to_="f5", turn=Color.WHITE))))
turns.append((STATE4, Action(from_= "f2", to_="e2", turn=Color.BLACK), heuristic(STATE1,  Action(from_= "f2", to_="e2", turn=Color.BLACK))))
his = History(Match(1, p1, p2, turns))
env = Environment(Board(strp_state(STATE5)), STATE5, his)