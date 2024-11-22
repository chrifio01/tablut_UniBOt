"""
This module provides utility functions for the Tablut game environment.

Functions:
    state_to_tensor(state: State, player_color: Color) -> np.ndarray:
        Convert the game state to a tensor representation suitable for DQN model input.
"""

import numpy as np
from shared.utils import State, StateFeaturizer, Color

def state_to_tensor(state: State, player_color: Color):
    """
    Convert the game state to a tensor representation suitable for model input.

    Args:
        state (State): The current state of the game.
        player_color (Color): The color of the player for whom the state is being featurized.

    Returns:
        np.ndarray: A tensor representation of the game state.
    """
    featurized_state = StateFeaturizer.generate_input(state, player_color)
    flattened_board_input = featurized_state.board_input.flatten()
    return np.concatenate([
        flattened_board_input,
        featurized_state.turn_input,
        featurized_state.white_input,
        featurized_state.black_input,
    ])