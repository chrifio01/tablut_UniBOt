import numpy as np

from shared.utils import State, StateFeaturizer, Color

def state_to_tensor(state: State, player_color: Color):
    featurized_state = StateFeaturizer.generate_input(state, player_color)
    flattened_board_input = featurized_state.board_input.flatten()
    return np.concatenate([
        flattened_board_input,
        featurized_state.turn_input,
        featurized_state.white_input,
        featurized_state.black_input,
    ])