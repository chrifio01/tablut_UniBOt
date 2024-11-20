import numpy as np

from shared.loggers import logger
from shared.utils.game_utils import Board, Action, Turn
from shared.utils.env_utils import State, black_win_con
from shared.consts import WIN_TILES
from shared.history import History
from shared.move_checker import MoveChecker
from shared.heuristic import heuristic


class Environment:
    board: Board
    currentState: State
    historyUpdater: History

    def __init__(self, board: Board, currentState: State, historyUpdater: History):
        self.board = board
        self.currentState = currentState
        self.historyUpdater = historyUpdater


    def is_it_a_tie(self, match_id: int) -> bool:

        turns = self.historyUpdater.matches[match_id].turns

        if len(turns) < 4:
            return False

        for i in range(len(turns) - 3):
            state1 = turns[i][0].board.pieces
            state2 = turns[i + 3][0].board.pieces

            if np.array_equal(state1, state2):
                return True

        return False


    def did_black_win(self) -> bool:
        if black_win_con(self.board, self.board.king_pos()) == 4:
            return True
        if self.currentState.turn == Turn.WHITE_TURN:
            if not MoveChecker.get_possible_moves(self.currentState):
                return True
        return False

    def did_white_win(self) -> bool:
        if self.board.king_pos() in WIN_TILES:
            return True
        if self.currentState.turn == Turn.BLACK_TURN:
            if not MoveChecker.get_possible_moves(self.currentState):
                return True
        return False

    def get_winnner(self):
        if self.did_black_win():
            return Turn.BLACK_WIN
        if self.did_white_win():
            return Turn.WHITE_WIN
        return None

    def calculate_rewards(self, match_id: int):
        turns = self.historyUpdater.matches[match_id].turns
        filtered_turns = list(filter(lambda tupla: tupla[0] == self.currentState, turns))
        if not filtered_turns:
            raise ValueError("Current state not found in turns.")
        action = filtered_turns[0][1]
        return heuristic(self.currentState, action)

    def update_state(self, move: Action):
        self.board.update_pieces(move)
        pass
