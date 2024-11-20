import os

import numpy as np

from environment.tablut import Environment
from shared.history import History, Match
from shared.random_player import RandomPlayer
from shared.utils import strp_color, strp_state, Color, Action, Turn
from shared.consts import INITIAL_STATE
from shared.loggers import logger
from connectors.client import Client
from shared.utils.game_utils import Board, strp_board

if __name__ == '__main__':

    """try:
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
    """


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
    "OOOOWOOOO\n"
    "BOOOWOOOB\n"
    "BBWWKWWBB\n"
    "BOOOWOOOB\n"
    "OOOOWOOOO\n"
    "OOOOOBOOO\n"
    "OOOBBBOOO\n"
    "-\n"
    "W"
)

STATE3 = (
    "OOOBBBOOO\n"
    "OOOOBOOOO\n"
    "OOOOWOOOO\n"
    "BOOOWOOOB\n"
    "BBWWKWWBB\n"
    "BOOOWOOOB\n"
    "OOOOWOOOO\n"
    "OOOOOOBOO\n"
    "OOOBBBOOO\n"
    "-\n"
    "W"
)


# Initialize players
p1 = RandomPlayer(color=Color.WHITE, initial_state=strp_state(STATE1))
p2 = RandomPlayer(color=Color.BLACK, initial_state=strp_state(STATE1))

# Initialize history with the same state twice
turns = [
    (strp_state(STATE1), Action(from_="f5", to_="f7", turn=Turn.WHITE_TURN), 0.0),
    (strp_state(STATE2), Action(from_="f5", to_="f7", turn=Turn.BLACK_TURN), 0.0),
    (strp_state(STATE3), Action(from_="f5", to_="f7", turn=Turn.WHITE_TURN), 0.0),
    (strp_state(STATE1), Action(from_="f5", to_="f7", turn=Turn.BLACK_TURN), 0.0)
]

# Create history and environment
his = History(matches={1: Match(match_id=1, white_player=p1, black_player=p2, turns=turns, outcome=None)})
env = Environment(board=Board(strp_state(STATE1).board.pieces), currentState=strp_state(STATE1), historyUpdater=his)

# Check if it is a tie
is_tie = env.is_it_a_tie(match_id=1)
print(f"Is it a tie? {is_tie}")