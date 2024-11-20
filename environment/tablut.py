import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import ArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
from shared.utils.game_utils import Board, Action, Turn
from shared.utils.env_utils import State, black_win_con
from shared.consts import WIN_TILES
from shared.history import History
from shared.move_checker import MoveChecker
from shared.heuristic import heuristic


class Environment(PyEnvironment):
    def __init__(self, board: Board, current_state: State, history: History):
        super().__init__()
        self.board = board
        self.currentState = current_state
        self.history = history
        self._episode_ended = False

    def action_spec(self):
        return ArraySpec(
            shape=(2,), dtype=np.int32, name='action')

    def observation_spec(self):
        return ArraySpec(
            shape=self.board.pieces.shape, dtype=np.int32, name='observation')

    def _reset(self) -> TimeStep:
        self.board.reset()
        self.currentState.reset()
        self._episode_ended = False
        return ts.restart(self.board.pieces)

    def _step(self, action: Action) -> TimeStep:
        if self._episode_ended:
            return self.reset()

        """# Apply the action and update the board
        self.board.update_pieces(action)
        self.currentState.update(action)

        # Check for termination conditions
        if self.did_black_win():
            self._episode_ended = True
            return ts.termination(self.board.pieces, reward=1.0)  # Black wins
        elif self.did_white_win():
            self._episode_ended = True
            return ts.termination(self.board.pieces, reward=-1.0)  # White wins

        # Check for a tie
        if self.is_it_a_tie(match_id=self.history.current_match_id):
            self._episode_ended = True
            return ts.termination(self.board.pieces, reward=0.0)  # Tie"""

        # Continue the episode
        return ts.transition(self.board.pieces, reward=0.0, discount=1.0)

    def is_it_a_tie(self, match_id: int) -> bool:
        turns = self.history.matches[match_id].turns
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

    def get_winner(self):
        if self.did_black_win():
            return Turn.BLACK_WIN
        if self.did_white_win():
            return Turn.WHITE_WIN
        return None

    def calculate_rewards(self, match_id: int):
        turns = self.history.matches[match_id].turns
        if not turns:
            raise ValueError("No turns found for the match.")
        last_state, last_action, _ = turns[-1]
        if np.array_equal(self.currentState.board.pieces, last_state.board.pieces):
            return heuristic(self.currentState, last_action)
        else:
            raise ValueError("Current state does not match the last state in history.")

    def update_state(self, move: Action):
        self.board.update_pieces(move)
