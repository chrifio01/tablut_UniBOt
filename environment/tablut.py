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

from typing import Tuple
from datetime import datetime
import random
import copy

import numpy as np
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import ArraySpec, array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.environments import tf_py_environment

from shared.utils import Action, Turn, State, black_win_con, strp_state, AbstractPlayer, Color, winner_color
from shared.consts import WIN_TILES, INITIAL_STATE, WIN_REWARD, LOSS_REWARD, DRAW_REWARD, INVALID_ACTION_PUNISHMENT
from shared.history import History, Match
from shared.move_checker import MoveChecker
from shared.exceptions import InvalidAction
from shared.heuristic import heuristic
from shared.random_player import RandomPlayer
from shared.loggers import env_logger
from environment.utils import ActionDecoder
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
        current_state: State = strp_state(INITIAL_STATE),
        *,
        history: History = History(matches={}),
        trainer: AbstractPlayer,
        observation_spec_shape: Tuple[int, int],
        action_spec_shape: Tuple[int, int],
        discount_factor: float,
        standard_dtype: np.dtype = np.float32,
        reward_function=None,
        opponent=None,
        action_min: int = 0,
        action_max: int = 399
    ):
        super().__init__()
        # Game and trainer settings
        env_logger.debug("Initializing environment...")
        self.current_state = current_state
        env_logger.debug(self.current_state.board)
        self.history = history
        self._trainer = trainer
        if trainer.color and opponent:
            self._opponent = opponent or self._init_opponent()
            self._opponent._color = Color.WHITE if self._trainer.color == Color.BLACK else Color.BLACK
        else:
            self._opponent = self._init_opponent()
        self.reward_function = reward_function or heuristic
        self.state_frequency = {}
        if self._trainer.color is None:
            self._set_trainer_color()
        env_logger.debug("Trainer color: %s\tOpponent color: %s", self._trainer.color, self._opponent.color)

        # Environment configuration
        self._observation_spec_shape = observation_spec_shape
        env_logger.debug("observation_spec shape: %s", self._observation_spec_shape)
        self._action_spec_shape = action_spec_shape
        env_logger.debug("action_spec shape: %s", self._action_spec_shape)
        self._standard_dtype = standard_dtype
        self._discount_factor = discount_factor
        env_logger.debug("discount_factor: %s", self._discount_factor)
        self._action_min = action_min
        self._action_max = action_max

        # Auxiliary variables
        self._episode_ended = False
        self._current_match_id = self._create_match_id()
        env_logger.debug("Match ID: %s", self._current_match_id)
        env_logger.debug("Environment initialized.")
        env_logger.debug("Initializing match...")
        self._initialize_match()

    def get_info(self):
        """
        Returns information about the current state of the environment.
        """
        return {
            "current_state": self.current_state,
            "history": self.history,
            "trainer": self._trainer,
            "opponent": self._opponent,
            "episode_ended": self._episode_ended,
            "current_match_id": self._current_match_id,
            "observation_spec_shape": self._observation_spec_shape,
            "action_spec_shape": self._action_spec_shape,
            "discount_factor": self._discount_factor,
            "state_frequency": self.state_frequency,
            "reward_function": self.reward_function
        }

    def get_state(self) -> State:
        """
        Returns the current state of the environment.
        """
        return self.current_state

    def set_state(self, state):
        """
        Sets the state of the environment.
        """
        self.current_state = state
        
    def to_TFPy(self):
        """
        Converts the environment to a TFPyEnvironment.
        """
        return tf_py_environment.TFPyEnvironment(self)

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
            white_player=white_player.name, black_player=black_player.name, outcome=None, turns=[]
        )
        env_logger.debug("Initialized history match with id: %s", self._current_match_id)
        env_logger.debug("Trainer color: %s\tOpponent color: %s", self._trainer.color, self._opponent.color)
        env_logger.debug("First to move: %s", self._trainer.name if self._trainer.color == Color.WHITE else self._opponent.name)
        # if opponent is the white player, let him perform the move
        if self._opponent.color == Color.WHITE:
            opponent_action = self._perform_opponent_turn()
            self.current_state.turn = self._get_outcome()
            env_logger.debug("Outcome: %s", self.current_state.turn)
            if self.current_state.turn is None:
                env_logger.debug("Switching turn...")
                self._switch_and_validate_turn(opponent_action)
            else:
                self._end_match()

    def action_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=self._action_min, maximum=self._action_max, name='action')

    def observation_spec(self):
        return ArraySpec(
            shape=self._observation_spec_shape, dtype=self._standard_dtype, name='observation')

    def _reset(self) -> TimeStep:
        """Reset the environment to the initial state."""
        env_logger.debug("Resetting environment...")
        self.current_state = strp_state(INITIAL_STATE)
        self._episode_ended = False
        self._current_match_id = self._create_match_id()
        self._opponent.color = random.choice([Color.BLACK, Color.WHITE])
        self.state_frequency.clear()
        self._set_trainer_color()
        self._initialize_match()
        return ts.restart(state_to_tensor(self.current_state, self._trainer.color))

    def _step(self, action: int) -> TimeStep:
        """Advance the environment by one step."""
        env_logger.debug("Episode ended: %s", self._episode_ended)
        if self._episode_ended:
            return self._reset()

        decoded_action = ActionDecoder.decode(action, self.current_state)
        # Update state and get trainer's reward
        env_logger.debug("Updating state with action: %s\n", decoded_action)
        trainer_reward = self._update_state(decoded_action)

        # Check termination conditions
        env_logger.debug("Checking termination condition...")
        if self._episode_ended:
            env_logger.debug("Episode ended")
            # check for invalid action
            if trainer_reward == INVALID_ACTION_PUNISHMENT:
                final_reward = INVALID_ACTION_PUNISHMENT
            else:
                final_reward = self._assign_termination_reward()
            env_logger.debug("Final reward: %s", final_reward)
            return ts.termination(
                state_to_tensor(self.current_state, self._trainer.color), reward=final_reward
            )

        env_logger.debug("Continuing with state:\n%s\tReward: %s", self.current_state, trainer_reward)

        # Continue the episode
        return ts.transition(
            state_to_tensor(self.current_state, self._trainer.color),
            reward=trainer_reward,
            discount=tf.constant(self._discount_factor, dtype=tf.float32)
        )

    def _is_it_a_tie(self) -> bool:
        """Check if the current state is a tie."""
        current_hash = hash(self.current_state.board.pieces.tobytes())
        self.state_frequency[current_hash] = self.state_frequency.get(current_hash, 0) + 1
        return self.state_frequency[current_hash] >= 2

    def _did_black_win(self) -> bool:
        """Check if black has won."""
        if black_win_con(self.current_state.board, self.current_state.board.king_pos()):
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
        env_logger.debug("Calculating outcome...")
        if self._did_black_win():
            return Turn.BLACK_WIN
        if self._did_white_win():
            return Turn.WHITE_WIN
        if self._is_it_a_tie():
            return Turn.DRAW
        return None

    def _calculate_rewards(self, current_state: State, action_performed: Action):
        """Calculate rewards based on the current state and action."""
        if action_performed is None:
            return INVALID_ACTION_PUNISHMENT
        return self.reward_function(current_state, action_performed)

    def _update_history(self, match_id: str, state: State, action=None, reward=None):
        """Update the match history."""
        white_player = self._opponent if self._opponent.color == Color.WHITE else self._trainer
        black_player = self._opponent if self._opponent.color == Color.BLACK else self._trainer
        env_logger.debug("Updating history with match id: %s...", match_id)
        self.history.update_history(match_id, white_player.name, black_player.name, copy.deepcopy(state), action, reward)
        env_logger.debug("History updated:\n%s", self.history.matches[match_id])

    def _update_state(self, move: Action):
        """Update the state with a given move."""
        reward = self._calculate_rewards(copy.deepcopy(self.current_state), move)

        if reward != INVALID_ACTION_PUNISHMENT:
            env_logger.debug("Calculated reward: %s", reward)
            self._update_history(self._current_match_id, self.current_state, move, reward)
            self.current_state.board.update_pieces(move)
            env_logger.debug("Updated Board:\n%s", self.current_state.board)
            self._handle_turn_and_outcome(move)
            return reward

        assert reward == INVALID_ACTION_PUNISHMENT
        env_logger.debug("Invalid move by the Trainer: \n%s", move)
        self._update_history(self._current_match_id, self.current_state, move, reward)
        self._episode_ended = True  # no reward assigned cause it was player fault
        self.current_state.turn = Turn.BLACK_WIN if self._trainer.color.value == Turn.WHITE_TURN.value else Turn.WHITE_WIN
        self.history.set_outcome(self._current_match_id, self.current_state.turn)
        return reward

    def _handle_turn_and_outcome(self, move: Action):
        """Handle the turn progression and match outcome."""
        env_logger.debug("Handling turn...")
        self.current_state.turn = self._get_outcome()
        env_logger.debug("Outcome: %s", self.current_state.turn)
        if self.current_state.turn is None:
            self._switch_and_validate_turn(move)
            opponent_action = self._perform_opponent_turn()
            self.current_state.turn = self._get_outcome()
            env_logger.debug("Outcome: %s", self.current_state.turn)
            if self.current_state.turn is None:
                self._switch_and_validate_turn(opponent_action)
            else:
                self._end_match()
        else:
            self._end_match()

    def _switch_and_validate_turn(self, move: Action):
        """Switch and validate the turn."""
        self.current_state.turn = (
            Turn.BLACK_TURN if move.turn == Turn.WHITE_TURN else Turn.WHITE_TURN
        )
        env_logger.debug("It's %s turn", self._trainer.name if self.current_state.turn.value == self._trainer.color.value else self._opponent.name)

    def _perform_opponent_turn(self):
        """Handle the opponent's turn."""
        opponent_action = self._opponent.fit(self.current_state)
        try:
            MoveChecker.is_valid_move(self.current_state, opponent_action)
            env_logger.debug("Opponent move:\n%s",opponent_action)
            self._update_history(self._current_match_id, self.current_state, opponent_action, None)
            self.current_state.board.update_pieces(opponent_action)
            env_logger.debug("Updated Board:\n%s", self.current_state.board)
            return opponent_action
        except InvalidAction:
            env_logger.debug("Opponent performed an invalid action: \n%s", opponent_action)
            self._update_history(self._current_match_id, self.current_state, opponent_action, None)
            self._reset()

    def _end_match(self):
        """Handle the end of the match."""
        env_logger.debug("Ending the match...")
        self._episode_ended = True
        final_reward = self._assign_termination_reward()
        env_logger.info("Final reward: %s", final_reward)
        self.history.set_outcome(self._current_match_id, self.current_state.turn)
        self._update_history(self._current_match_id, self.current_state, None, final_reward)

    def _assign_termination_reward(self) -> float:
        """Assign the final reward based on the match outcome."""
        outcome_color = winner_color(self.current_state.turn)
        env_logger.debug("Outcome color: %s", outcome_color)
        if outcome_color == self._opponent.color:
            return float(LOSS_REWARD)
        if outcome_color == self._trainer.color:
            return float(WIN_REWARD)
        return float(DRAW_REWARD)
