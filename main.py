import unittest
import numpy as np

from shared.consts import INITIAL_STATE
from shared.heuristic import _white_heuristic, _black_heuristic, heuristic
from shared.utils import strp_state
from shared.utils.game_utils import Board, Action, Color, Piece, Turn
from shared.utils.env_utils import State

class TestHeuristicMethods(unittest.TestCase):

    def setUp(self):
        # Setup a sample board state for testing
        self.initial_board_state = np.array([
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.KING, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
            [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY]
        ])
        self.board = Board(self.initial_board_state)

    def test_white_heuristic(self):
        fitness = _white_heuristic(self.board)
        self.assertIsInstance(fitness, float)

    def test_black_heuristic(self):
        fitness = _black_heuristic(self.board)
        self.assertIsInstance(fitness, float)

    def test_heuristic(self):
        state = strp_state(INITIAL_STATE)
        move = Action(from_="e3", to_="f3", turn=Turn.WHITE_TURN)
        fitness = heuristic(state, move)
        self.assertIsInstance(fitness, float)

if __name__ == '__main__':
    unittest.main()