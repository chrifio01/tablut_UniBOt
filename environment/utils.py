"""
This module provides utility functions and classes for the Tablut game environment, 
including state representation and action decoding.

Functions:
    state_to_tensor(state: State, player_color: Color) -> np.ndarray:
        Convert the game state to a tensor representation suitable for DQN model input.

Classes:
    ActionDecoder:
        Decodes an action tensor produced by a DQN into a valid Tablut action.
"""
from typing import Tuple
import tensorflow as tf
from tf_agents.policies import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep

import numpy as np
from shared.utils import State, StateFeaturizer, Color, Action, Piece, strf_square, strp_square
from shared.consts import DEFENDER_NUM, ATTACKER_NUM
from shared.utils.env_utils import StateDecoder
from shared import MoveChecker

def state_to_tensor(state: State, player_color: Color) -> np.ndarray:
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
        featurized_state.turn_input
    ]).astype(np.float32)
    
class ActionDecoder:
    """
    Decodes an action tensor produced by a DQN model into a valid Tablut action.

    Methods:
        _get_piece_type(action_column_index: int) -> Piece:
            Determines the type of piece (King, Defender, or Attacker) based on the column index.
        
        _num_pieces(piece: Piece) -> int:
            Returns the number of pieces of the specified type on the board.

        _get_destination_coordinates(action_index: Tuple[int, int], 
                                     moving_pawn_coords: Tuple[int, int], 
                                     state: State) -> Tuple[int, int]:
            Calculates the destination coordinates of a pawn based on its starting position 
            and the specified move.

        _get_moving_pawn_coordinates(action_index: Tuple[int, int], state: State) -> Tuple[int, int]:
            Determines the starting coordinates of the pawn associated with the specified 
            action index.

        decode(action_tensor: np.ndarray, state: State) -> Action:
            Converts a DQN-generated action tensor into a valid Tablut action.
    """
    
    @staticmethod
    def _get_piece_type(action_column_index: int) -> Piece:
        """
        Determine the type of piece (King, Defender, or Attacker) based on the column index.

        Args:
            action_column_index (int): The column index of the action tensor.

        Returns:
            Piece: The corresponding piece type.
        """
        if action_column_index == 0:
            return Piece.KING
        if action_column_index in range(1, 9):
            return Piece.DEFENDER
        if action_column_index in range(9, 25):
            return Piece.ATTACKER
        raise IndexError("Action_column_index out of range")
    
    @staticmethod
    def _num_pieces(piece: Piece) -> int:
        """
        Get the number of pieces of a specific type.

        Args:
            piece (Piece): The type of the piece (King, Defender, Attacker).

        Returns:
            int: The number of pieces of the specified type.
        """
        if piece == Piece.ATTACKER:
            return ATTACKER_NUM
        if piece == Piece.DEFENDER:
            return DEFENDER_NUM
        raise ValueError("Invalid piece type")
    
    @staticmethod
    def _get_destination_coordinates(action_index: Tuple[int, int], 
                                     moving_pawn_coords: Tuple[int, int], 
                                     state: State) -> Tuple[int, int]:
        """
        Calculate the destination coordinates for the pawn based on the action tensor and the current board state.

        Args:
            action_index (Tuple[int, int]): The index of the action in the tensor.
            moving_pawn_coords (Tuple[int, int]): The coordinates of the moving pawn.
            state (State): The current game state.

        Returns:
            Tuple[int, int]: The destination coordinates of the move.
        """
        row, col = moving_pawn_coords
        move_index = action_index[1]  # Second index specifies the move
        pieces = state.board.pieces  # Board representation for obstacle checking

        # Generate valid moves
        valid_moves = []

        # Vertical moves (0-7)
        for r in range(row):  # Upward moves
            valid_moves.append((r, col))
        for r in range(row + 1, pieces.shape[0]):  # Downward moves
            valid_moves.append((r, col))

        # Horizontal moves (8-15)
        for c in range(col):  # Leftward moves
            valid_moves.append((row, c))
        for c in range(col + 1, pieces.shape[1]):  # Rightward moves
            valid_moves.append((row, c))

        # Exclude the starting position and ensure index bounds
        if moving_pawn_coords in valid_moves:
            valid_moves.remove(moving_pawn_coords)
        if move_index >= len(valid_moves):
            raise ValueError("Move index out of bounds for the available valid moves.")

        # Return the move corresponding to the move index
        return valid_moves[move_index]
    
    @staticmethod
    def _get_moving_pawn_coordinates(action_index: Tuple[int, int], state: State) -> Tuple[int, int]:
        """
        Find the starting coordinates of the moving pawn based on the action tensor index.

        Args:
            action_index (Tuple[int, int]): The index of the action in the tensor.
            state (State): The current game state.

        Returns:
            Tuple[int, int]: The coordinates of the moving pawn.
        """
        piece_type = ActionDecoder._get_piece_type(action_index[0])
        target_indices = np.argwhere(state.board.pieces == piece_type)

        if len(target_indices) == 0:
            raise ValueError(f"No pieces of type {piece_type} found on the board.")

        # Sort pieces by Manhattan distance from (0, 0)
        distances = np.abs(target_indices - np.array([0, 0])).sum(axis=1)
        sorted_indices = target_indices[np.argsort(distances)]

        # Determine the rank of the selected piece
        if piece_type == Piece.DEFENDER:
            piece_rank = action_index[0] - 1  # King is index 0
        elif piece_type == Piece.ATTACKER:
            piece_rank = action_index[0] - ActionDecoder._num_pieces(Piece.DEFENDER) - 1  # Offset for defenders and king
        elif piece_type == Piece.KING:
            piece_rank = 0
        else:
            raise ValueError("Invalid piece type")

        if piece_rank >= len(sorted_indices):
            raise ValueError(f"Piece rank {piece_rank} exceeds available pieces of type {piece_type}.")

        return tuple(sorted_indices[int(piece_rank)])
    
    @staticmethod
    def decode(action_index: int, state: State) -> Action:
        """
        Decode the flattened action tensor into a valid Tablut action.

        Args:
            action_index (int): The index of the move..
            state (State): The current game state.

        Returns:
            Action: The decoded action object.
        """
        # Map flat index to 2D action indices: (action_column_index, move_index)
        action_column_index = action_index // 16
        move_index = action_index % 16
        
        try:
            # Get the starting coordinates of the pawn being moved
            from_tuple = ActionDecoder._get_moving_pawn_coordinates((action_column_index, move_index), state)

            # Get the destination coordinates for the pawn
            to_tuple = ActionDecoder._get_destination_coordinates((action_column_index, move_index), from_tuple, state)

            # Retrieve the turn information from the state
            turn = state.turn

            # Return the constructed Action object
            return Action(from_=strf_square(from_tuple), to_=strf_square(to_tuple), turn=turn)
        except:
            return None
        
class ActionEncoder:
    """
    Encodes a Tablut action into a tensor format suitable for input into a DQN model.

    Methods:
        encode(action: Action, state: State) -> int:
            Converts a valid Tablut action into an action index for the model.
    """

    @staticmethod
    def encode(action: Action, state: State) -> int:
        """
        Encodes the action into an index suitable for model input.

        Args:
            action (Action): The action to encode.
            state (State): The current game state.

        Returns:
            int: The encoded action index.
        """
        from_coordinates = strp_square(action.from_)
        to_coordinates = strp_square(action.to_)

        # Get the piece type from the action
        piece_type = state.board.get_piece(from_coordinates)
        piece_index = ActionEncoder._get_piece_index(from_coordinates, piece_type, state)

        # Get the move index based on destination
        move_index = ActionEncoder._get_move_index(from_coordinates, to_coordinates, state)

        # Calculate the final action index
        action_index = piece_index * 16 + move_index
        return action_index

    @staticmethod
    def _get_piece_index(coordinates: Tuple[int, int], piece_type: Piece, state: State) -> int:
        """
        Gets the index of the piece of the specified type based on its coordinates.

        Args:
            coordinates (Tuple[int, int]): The coordinates of the piece.
            piece_type (Piece): The type of the piece (King, Defender, Attacker).
            state (State): The current game state.

        Returns:
            int: The index of the piece.
        """
        target_indices = np.argwhere(state.board.pieces == piece_type)
        
        # Calculate distances from (0, 0) to sort by proximity
        distances = np.linalg.norm(target_indices - np.array([0, 0]), axis=1)
        sorted_indices = target_indices[np.argsort(distances)]

        # Determine the index based on the piece type and its rank in sorted order
        if piece_type == Piece.KING:
            return 0  # King is always at index 0
        elif piece_type == Piece.DEFENDER:
            # Find the rank of the defender piece based on its position
            piece_rank = np.where((coordinates[0], coordinates[1]) == sorted_indices)[0][0]
            return piece_rank + 1  # Indices 1-8 for defenders
        elif piece_type == Piece.ATTACKER:
            # Find the rank of the attacker piece based on its position
            piece_rank = np.where((coordinates[0], coordinates[1]) == sorted_indices)[0][0]
            return piece_rank + 9  # Indices 9-24 for attackers
        else:
            raise ValueError("Invalid piece type")

    @staticmethod
    def _get_move_index(from_coordinates: Tuple[int, int], to_coordinates: Tuple[int, int], state: State) -> int:
        """
        Gets the move index based on the destination coordinates.

        Args:
            from_coordinates (Tuple[int, int]): The starting coordinates of the piece.
            to_coordinates (Tuple[int, int]): The destination coordinates of the move.
            state (State): The current game state.

        Returns:
            int: The index of the move.
        """
        row, col = from_coordinates
        pieces = state.board.pieces

        valid_moves = []
        
        # Collect all valid moves in the same way as in MoveChecker
        # Vertical moves
        for r in range(row):  # Upward moves
            valid_moves.append((r, col))
        for r in range(row + 1, pieces.shape[0]):  # Downward moves
            valid_moves.append((r, col))

        # Horizontal moves
        for c in range(col):  # Leftward moves
            valid_moves.append((row, c))
        for c in range(col + 1, pieces.shape[1]):  # Rightward moves
            valid_moves.append((row, c))

        # Return the index corresponding to the destination coordinates
        if to_coordinates in valid_moves:
            return valid_moves.index(to_coordinates)
        else:
            raise ValueError("Destination coordinates are not a valid move.")

class TablutCustomTFPolicy(TFPolicy):
    """
    A custom TFPolicy for Tablut that uses MoveChecker to validate moves
    and selects a random valid move as the action.
    """

    def __init__(self, time_step_spec, action_spec, name=None):
        """
        Initializes the policy.

        Args:
            time_step_spec: Specification of the input time steps.
            action_spec: Specification of the actions.
            move_checker: Instance of the MoveChecker class for move validation.
            state_adapter: Function to adapt the observation to a MoveChecker-compatible state.
            name: Name of the policy.
        """
        super().__init__(time_step_spec, action_spec, name=name)

    def _variables(self):
        """Return the trainable variables (none for this random policy)."""
        return []

    def _distribution(self, time_step, policy_state):
        """Return the distribution over actions (not used for this policy)."""
        raise NotImplementedError("This policy generates actions directly, not distributions.")

    def _action(self, time_step: TimeStep, policy_state, seed=None):
        """
        Generate a random valid action for the current time step.

        Args:
            time_step (TimeStep): Current time step containing the observation.
            policy_state: Current state of the policy (unused for random policy).
            seed: Random seed for reproducibility (optional).

        Returns:
            PolicyStep: The chosen action and updated policy state.
        """
        # Convert observation to a MoveChecker-compatible state
        state = StateDecoder.decode(time_step.observation)

        # Generate all valid moves using MoveChecker
        valid_moves = list(MoveChecker.gen_possible_moves(state))

        if not valid_moves:
            raise ValueError("No valid moves available for the current state.")

        # Randomly select one of the valid moves
        np_random = np.random.default_rng(seed)
        selected_move = np_random.choice(valid_moves)

        # Encode the move into an index compatible with the action_spec
        action_index = ActionEncoder.encode(selected_move, state)

        # Return the action as a PolicyStep
        return PolicyStep(action=np.array([action_index], dtype=np.int32), state=policy_state)
