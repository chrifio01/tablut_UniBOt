"""
This module defines the `Environment` class for implementing a reinforcement learning (RL) environment 
for the Ashton Tablut board game. The environment is built to be compatible with TensorFlow Agents (TF-Agents) 
and supports DQN-based RL.

Key Features:
- Tracks match history and outcomes.
- Implements game rules and turn progression for both players.
- Handles match termination conditions and rewards assignment.
- Allows integration of custom reward functions and opponent logic.

Classes:
- Environment: The primary RL environment class for Ashton Tablut.
"""

from typing import Tuple, Union
from datetime import datetime
import random
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import ArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
from shared.utils import Action, Turn, State, black_win_con, strp_state, AbstractPlayer, Color, winner_color
from shared.consts import WIN_TILES, INITIAL_STATE, WIN_REWARD, LOSS_REWARD, DRAW_REWARD
from shared.history import History, Match
from shared.move_checker import MoveChecker
from shared.heuristic import heuristic
from shared.random_player import RandomPlayer
from .utils import state_to_tensor


class Environment(PyEnvironment):
    """
    A reinforcement learning environment for the Ashton Tablut game, built for TF-Agents compatibility.

    The `Environment` class simulates matches between a trainer (RL agent) and an opponent. It manages
    the game state, history, and rewards, while adhering to the rules of Ashton Tablut.

    Attributes:
        current_state (State): The current state of the board and game.
        history (History): Tracks all moves and outcomes for matches.
        _trainer (AbstractPlayer): The RL agent playing as one of the colors.
        _opponent (AbstractPlayer): The opponent player, defaulting to a random strategy.
        reward_function (callable): A function to calculate rewards, defaulting to a heuristic.
        _episode_ended (bool): Tracks whether the current match has ended.
        _current_match_id (str): Unique identifier for the ongoing match.
        _observation_spec_shape (tuple): Shape of the observation tensor for the agent.
        _action_spec_shape (tuple): Shape of the action tensor for the agent.
        _standard_dtype (np.dtype): Data type for observations and actions.
        _discount_factor (float): Discount factor for rewards.

    Methods:
        action_spec(): Returns the action specification for the environment.
        observation_spec(): Returns the observation specification for the environment.
        _reset(): Resets the environment to the initial state.
        _step(action): Advances the environment by one step, updating the state and calculating rewards.
        _is_it_a_tie(): Checks if the game has ended in a tie.
        _did_black_win(): Checks if black has won.
        _did_white_win(): Checks if white has won.
        _get_outcome(): Determines the current match outcome.
        _calculate_rewards(current_state, action_performed): Computes rewards for a given state and action.
        _update_history(match_id, state, action, reward): Updates the match history.
        _update_state(move): Updates the game state based on the trainer's move.
    """
    
    def __init__(
        self,
        history: History,
        current_state: State = strp_state(INITIAL_STATE),
        *,
        trainer: AbstractPlayer,
        observation_spec_shape: Tuple[int, int],
        action_spec_shape: Tuple[int, int],
        discount_factor: float,
        standard_dtype: np.dtype = np.float16,
        reward_function=None,
        opponent=None
    ):
        super().__init__()
        # Game and trainer settings
        self.current_state = current_state
        self.history = history
        self._trainer = trainer
        self._opponent = opponent or self._init_opponent()
        self.reward_function = reward_function or heuristic
        self._set_trainer_color()

        # Environment configuration
        self._observation_spec_shape = observation_spec_shape
        self._action_spec_shape = action_spec_shape
        self._standard_dtype = standard_dtype
        self._discount_factor = discount_factor

        # Auxiliary variables
        self._episode_ended = False
        self._current_match_id = self._create_match_id()
        self._initialize_match()

    @staticmethod
    def _init_opponent():
        """Initialize a default random opponent."""
        opponent_color = random.choice([Color.BLACK, Color.WHITE])
        return RandomPlayer(opponent_color, strp_state(INITIAL_STATE))

    def _set_trainer_color(self):
        """Set trainer's color based on the opponent's color."""
        self._trainer.color = Color.BLACK if self._opponent.color == Color.WHITE else Color.WHITE

    def _create_match_id(self) -> str:
        """Generate a unique match ID based on the current timestamp."""
        return datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S")

    def _initialize_match(self):
        """Initialize a new match in the history."""
        if self._trainer.color == Color.WHITE:
            white_player = self._trainer
            black_player = self._opponent
        else:
            white_player = self._opponent
            black_player = self._trainer
        self.history.matches[self._current_match_id] = Match(
            white_player=white_player, black_player=black_player, outcome=None, turns=[]
        )

    def action_spec(self):
        return ArraySpec(
            shape=self._action_spec_shape, dtype=self._standard_dtype, name='action')

    def observation_spec(self):
        return ArraySpec(
            shape=self._observation_spec_shape, dtype=self._standard_dtype, name='observation')

    def _reset(self) -> TimeStep:
        """Reset the environment to the initial state."""
        self.current_state = strp_state(INITIAL_STATE)
        self._episode_ended = False
        self._current_match_id = self._create_match_id()
        self._opponent.color = random.choice([Color.BLACK, Color.WHITE])
        self._set_trainer_color()
        self._initialize_match()
        return ts.restart(state_to_tensor(self.current_state, self._trainer.color))

    def _step(self, action: Action) -> TimeStep:
        """Advance the environment by one step."""
        if self._episode_ended:
            return self._reset()

        # Update state and get trainer's reward
        trainer_reward = self._update_state(action)

        # Check termination conditions
        if self._episode_ended:
            final_reward = self._assign_termination_reward()
            return ts.termination(
                state_to_tensor(self.current_state, self._trainer.color), reward=final_reward)

        # Continue the episode
        return ts.transition(
            state_to_tensor(self.current_state, self._trainer.color),
            reward=trainer_reward,
            discount=self._discount_factor
        )

    def _is_it_a_tie(self) -> bool:
        """Check if the current state is a tie."""
        current_hash = hash(self.current_state.board.pieces.tobytes())
        state_hashes = [hash(turn[0].board.pieces.tobytes()) for turn in self.history.matches[self._current_match_id].turns]
        return current_hash in state_hashes

    def _did_black_win(self) -> bool:
        """Check if black has won."""
        if black_win_con(self.current_state.board, self.current_state.board.king_pos()) == 4:
            return True
        if self.current_state.turn == Turn.WHITE_TURN:
            if not list(MoveChecker.gen_possible_moves(self.current_state)):
                return True
        return False

    def _did_white_win(self) -> bool:
        """Check if white has won."""
        if self.current_state.board.king_pos() in WIN_TILES:
            return True
        if self.current_state.turn == Turn.BLACK_TURN:
            if not list(MoveChecker.gen_possible_moves(self.current_state)):
                return True
        return False

    def _get_outcome(self):
        """Determine the outcome of the match."""
        if self._did_black_win():
            return Turn.BLACK_WIN
        if self._did_white_win():
            return Turn.WHITE_WIN
        if self._is_it_a_tie():
            return Turn.DRAW
        return None

    def _calculate_rewards(self, current_state: State, action_performed: Action):
        """Calculate rewards based on the current state and action."""
        return self.reward_function(current_state, action_performed)

    def _update_history(self, match_id: str, state: State, action=None, reward=None):
        """Update the match history."""
        self.history.matches[match_id].turns.append((state, action, reward))

    def _update_state(self, move: Action):
        """Update the state with a given move."""
        reward = self._calculate_rewards(self.current_state, move)
        self._update_history(self._current_match_id, self.current_state, move, reward)
        self.current_state.board.update_pieces(move)
        self._handle_turn_and_outcome(move)
        return reward

    def _handle_turn_and_outcome(self, move: Action):
        """Handle the turn progression and match outcome."""
        self.current_state.turn = self._get_outcome()
        if self.current_state.turn is None:
            self._switch_and_validate_turn(move)
            self._perform_opponent_turn()
            self.current_state.turn = self._get_outcome()
            if self.current_state.turn is None:
                self._switch_and_validate_turn(move)
            else:
                self._end_match()
        else:
            self._end_match()

    def _switch_and_validate_turn(self, move: Action):
        """Switch and validate the turn."""
        self.current_state.turn = (
            Turn.BLACK_TURN if move.turn == Turn.WHITE_TURN else Turn.WHITE_TURN
        )
        if self.current_state.turn.value != self._opponent.color.value:
            raise ValueError("Unexpected turn! It should be the opponent's turn.")

    def _perform_opponent_turn(self):
        """Handle the opponent's turn."""
        opponent_action = self._opponent.fit(self.current_state)
        self._update_history(self._current_match_id, self.current_state, opponent_action, None)
        self.current_state.board.update_pieces(opponent_action)

    def _end_match(self):
        """Handle the end of the match."""
        self._episode_ended = True
        final_reward = self._assign_termination_reward()
        self._update_history(self._current_match_id, self.current_state, None, final_reward)
        self.history.matches[self._current_match_id].outcome = self.current_state.turn

    def _assign_termination_reward(self) -> float:
        """Assign the final reward based on the match outcome."""
        outcome_color = winner_color(self.current_state.turn)
        if outcome_color == self._opponent.color:
            return float(LOSS_REWARD)
        if outcome_color == self._trainer.color:
            return float(WIN_REWARD)
        return float(DRAW_REWARD)
