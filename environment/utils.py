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

import numpy as np
from shared.utils import State, StateFeaturizer, Color, Action, Piece, strf_square
from shared.consts import DEFENDER_NUM, ATTACKER_NUM

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
        featurized_state.turn_input,
        featurized_state.white_input,
        featurized_state.black_input,
    ]).astype(np.float16)

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

        return tuple(sorted_indices[piece_rank])

    @staticmethod
    def decode(flat_action_tensor: np.ndarray, state: State) -> Action:
        """
        Decode the flattened action tensor into a valid Tablut action.

        Args:
            flat_action_tensor (np.ndarray): The flattened action tensor of size 400.
            state (State): The current game state.

        Returns:
            Action: The decoded action object.
        """
        # Find the index of the maximum Q-value in the flattened action tensor
        flat_index = np.argmax(flat_action_tensor)

        # Map flat index to 2D action indices: (action_column_index, move_index)
        action_column_index = flat_index // 16
        move_index = flat_index % 16

        # Get the starting coordinates of the pawn being moved
        from_tuple = ActionDecoder._get_moving_pawn_coordinates((action_column_index, move_index), state)

        # Get the destination coordinates for the pawn
        to_tuple = ActionDecoder._get_destination_coordinates((action_column_index, move_index), from_tuple, state)

        # Retrieve the turn information from the state
        turn = state.turn

        # Return the constructed Action object
        return Action(from_=strf_square(from_tuple), to_=strf_square(to_tuple), turn=turn)