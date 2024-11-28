"""
This module defines the `State` model and related utility functions for parsing the game state in Tablut.

Classes:
    State: A Pydantic model representing the current state of the Tablut game, including the board
        configuration and turn information.

Functions:
    strp_state(state_str: str) -> Annotated[State, "The corresponding state from a string representation
        of the state sent from the server"]:
        Parses a server-provided string to create a `State` object, which includes the board's piece 
        configuration and the player's turn.

Usage Example:
    To parse a game state from a string:
        state_str = "OOOBBBOOO\nOOOOBOOOO\n... - WHITE"
        state = strp_state(state_str)
"""
from math import sqrt
from typing import Annotated
import numpy as np
from pydantic import BaseModel
from shared.consts import WEIGHTS, CAMPS
from .game_utils import Board, strp_board, Piece, strp_turn, parse_state_board, Turn, Color

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


__all__ = ['State', 'strp_state', 'state_decoder']


class State(BaseModel):
    """
    Model class representing the state of the game in Tablut.

    Attributes:
        board (Board): The current state of the game board, represented as a 2D array of `Piece` values.
        turn (Color): The player whose turn it currently is, represented as a `Color` enum.
    """
    board: Board
    turn: Turn

    class Config:
        """
            Allow arbitrary types for the model. This allows for more flexibility in parsing JSON objects.
        """
        arbitrary_types_allowed = True

    def __str__(self):
        return f"{self.board.__str__()}\n-\n{self.turn.value}"


def state_decoder(obj: dict):
    """
    Decodes JSON objects into `State` objects.

    Args:
        obj (dict): The JSON object to be decoded.

    Returns:
        State: A `State` object created from the provided JSON object.
    """
    if 'turn' in obj and 'board' in obj:
        turn = strp_turn(obj['turn'])
        board = parse_state_board(obj['board'])
        return State(board=board, turn=turn)
    return None


def strp_state(
        state_str: str
) -> Annotated[State, "The corresponding state from a string representation of the state"]:
    """
    Converts a server-provided string representation of the game state into a `State` object.

    Args:
        state_str (str): A string representing the state in the format of "<board layout> - <turn>",
            where "<board layout>" contains rows of pieces and "<turn>" specifies the current player.

    Returns:
        State: A `State` object representing the parsed game state.

    Raises:
        ValueError: If the provided `state_str` does not match the expected format for board and turn.
        IndexError: If there is an error in parsing due to an incomplete or malformed string.

    Example:
        state = strp_state("OOOBBBOOO\nOOOOBOOOO\n... - WHITE")
    """
    try:
        splitted_state_str = state_str.split('-')
        board_state_str = splitted_state_str[0].strip()
        turn_str = splitted_state_str[1].strip()

        pieces = strp_board(board_state_str)
        board = Board(pieces)
        board.pieces = pieces  # Set board configuration for non-initial states

        return State(board=board, turn=Turn(turn_str))
    except IndexError as e:
        raise ValueError("Invalid state format: missing board or turn information.") from e
    except ValueError as e:
        raise ValueError("Invalid state format: could not parse board or turn.") from e


############################################### Definition of the functions for the evaluation of the Fitness in the heuristic ###########################################################################


def king_distance_from_center(board: Board, king: tuple[int, int]):
    """
    Calculate de distance of the king from the center

    Args:
    a Board object
    The king coordinates as a tuple
    """
    return sqrt((king[0] - (board.height // 2)) ** 2 + (king[1] - (board.width // 2)) ** 2)



def king_surrounded(board: Board):
    """
    Return the number of sides in which the king is surrounded by an enemy (max(c) = 4)
    Return also a list with the blocked position around the king

    Args:
    Board object
    """
    king = board.king_pos()
    c = 0
    blocked_pos = []

    if king[0] + 1 >= board.height:
        c += 1
    elif board.get_piece((king[0] + 1, king[1])) in (Piece.ATTACKER, Piece.THRONE) or (king[0] + 1, king[1]) in CAMPS:
        c += 1
        blocked_pos.append((king[0] + 1, king[1]))
    if king[0] - 1 < 0:
        c += 1
    elif board.get_piece((king[0] - 1, king[1])) in (Piece.ATTACKER, Piece.THRONE)  or (king[0] - 1, king[1]) in CAMPS:
        c += 1
        blocked_pos.append((king[0] - 1, king[1]))
    if king[1] + 1 >= board.width:
        c += 1
    elif board.get_piece((king[0], king[1] + 1)) in (Piece.ATTACKER, Piece.THRONE) or (king[0], king[1] + 1) in CAMPS:
        c += 1
        blocked_pos.append((king[0], king[1] + 1))
    if king[1] - 1 < 0:
        c += 1
    elif board.get_piece((king[0], king[1] - 1)) in (Piece.ATTACKER, Piece.THRONE) or (king[0], king[1] - 1) in CAMPS:
        c += 1
        blocked_pos.append((king[0], king[1] - 1))

    return c, blocked_pos


def position_weight(king: tuple[int, int]):
    """
    Return a value depending on the position of the king on the board

    Args:
    Tuple with the king's coordinates
    """
    return WEIGHTS[king[0]][king[1]]


def pawns_around(board: Board, pawn: tuple, distance: int):
    """
    Returns the number of pawns around a given pawn within a certain distance (usually the king)

    Args:
    Board object, the coordinate of the target pawn as a tuple, the distance of the search from the target
    """
    x, y = pawn
    count = 0
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            if i == 0 and j == 0:
                continue
            if (x + i, y + j) in board.get_black_coordinates():
                count += 1
    return count


def piece_parser(piece: Piece) -> int:
    """
    Return the index of the boolean array (which represents the board) used as a input for the policy network of the DQN

    Arg:
    Piece object

    Example:
    If the piece given is the KING, the function will return 1
    The second array given as input will be the one displaying the position of the KING in the 9x9 board (index 1 means second element)
    """
    state_pieces = {Piece.DEFENDER: 0,
                    Piece.KING: 1,
                    Piece.ATTACKER: 2,
                    Piece.CAMPS: 3,
                    Piece.THRONE: 3}
    return state_pieces[piece]

class StateDecoder:
    """
    Decodes a tensor representation of a state into a State object and additional information.

    Methods:
        decode(state_tensor: np.ndarray, player_color: Color) -> Tuple[State, Color]:
            Decodes a tensor into a State object and the player's color.
    """

    @staticmethod
    def decode(state_tensor: np.ndarray) -> State:
        """
        Decode a tensor representation back into a State object.

        Args:
            state_tensor (np.ndarray): The tensor representation of the state.
            player_color (Color): The color of the player for whom the state was featurized.

        Returns:
            Tuple[State, Color]: The decoded State object and the player's color.
        """
        # Ensure the input is converted into a NumPy array
        state_tensor_l = state_tensor.tolist()[0]

        # Define sizes for slicing
        flattened_board_size = 4 * 9 * 9  # 324

        # Extract the board input and reshape
        board_input = np.array(state_tensor_l[:flattened_board_size]).reshape((4, 9, 9)).astype(bool)

        # Extract turn information
        turn_input = state_tensor_l[flattened_board_size:flattened_board_size + 1]
        is_white_turn = bool(turn_input[0])

        # Determine whose turn it is based on the input
        turn_color = Turn.WHITE_TURN if is_white_turn else Turn.BLACK_TURN

        # Reverse map the board representation
        board = np.full((9, 9), Piece.EMPTY, dtype=Piece)  # Initialize an empty board
        for i in range(9):
            for j in range(9):
                if board_input[piece_parser(Piece.KING)][i, j]:
                    board[i, j] = Piece.KING
                elif board_input[piece_parser(Piece.DEFENDER)][i, j]:
                    board[i, j] = Piece.DEFENDER
                elif board_input[piece_parser(Piece.ATTACKER)][i, j]:
                    board[i, j] = Piece.ATTACKER
            
        if board[9//2] [9//2] != Piece.KING:
            board[9//2][9//2] = Piece.THRONE

        # Create a State object using the reconstructed board
        decoded_board = Board(board)
        decoded_state = State(board=decoded_board, turn=turn_color)

        # Return the decoded state and player color
        return decoded_state
    
class FeaturizedState(BaseModel):
    """
    Model representing the featurized state of the Tablut game for input into the DQN.

    Attributes:
        board_input (np.ndarray): A 5x9x9 boolean array representing the positions of different pieces on the board.
        turn_input (np.ndarray): A boolean array indicating the current player's turn.
        white_input (np.ndarray): An array containing heuristic values for the white player.
        black_input (np.ndarray): An array containing heuristic values for the black player.
    """
    board_input: np.ndarray
    turn_input: np.ndarray
    white_input: np.ndarray
    black_input: np.ndarray

    class Config:
        """
        Configuration class for the Pydantic model.

        Attributes:
            arbitrary_types_allowed (bool): Allows the model to accept arbitrary types.
        """
        arbitrary_types_allowed = True

class StateFeaturizer:
    """
    Class representing the state given as input to the DQN.

    Methods:
        generate_input(): Generates the tensor input of the DQN from the position of the pieces, the turn and the points given from the black and white heuristics
    """

    @staticmethod
    def generate_input(state: State, player_color: Color) -> FeaturizedState:
        """
        Return the tensor representing the state which the DQN should receive as input to choose best action

        """
        position_layer = np.zeros((4, 9, 9), dtype=bool)
        for x, y in CAMPS:
            position_layer[piece_parser(Piece.CAMPS)][x, y] = 1
        position_layer[piece_parser(Piece.CAMPS)][4, 4] = 1

        board = state.board

        for i in range(board.height):
            for j in range(board.width):
                try:
                    position = (i, j)
                    piece = piece_parser(Piece(board.get_piece(position)))
                    position_layer[piece][i, j] = True
                except KeyError:
                    pass

        turn_layer = np.array([1 if player_color == Color.WHITE else 0], dtype=bool)
        
        w_heur_layer = np.array(
            [board.num_black(), board.num_white(), king_distance_from_center(board, board.king_pos()),
             king_surrounded(board)[0], position_weight(board.king_pos())])

        b_heur_layer = np.array(
            [board.num_black(), board.num_white(), pawns_around(board, board.king_pos(), 1)])

        return FeaturizedState(board_input=position_layer, turn_input=turn_layer, white_input=w_heur_layer, black_input=b_heur_layer)


def black_win_con(board: Board, king: tuple[int, int]) -> bool:
    """
    Determines if the Black player captures the King.

    Args:
        board (Board): The board object representing the game state.
        king (tuple[int, int]): The position of the King as a (row, column) tuple.

    Returns:
        bool: True if the Black player captures the King, False otherwise.
    """
    x, y = king
    adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    blockers = []  # List of what's blocking each side

    for pos in adjacent_positions:
        if 0 <= pos[0] < board.height and 0 <= pos[1] < board.width:
            piece = board.get_piece(pos)
            if piece in {Piece.ATTACKER, Piece.THRONE} or pos in CAMPS:
                blockers.append((pos, piece))

    # Condition 1: Surrounded by Attackers on all four sides
    if all(block[1] == Piece.ATTACKER for block in blockers) and len(blockers) == 4:
        return True

    # Condition 2: Adjacent to the Throne, blocked on the other three sides by Attackers
    throne_adjacent = any(block[1] == Piece.THRONE for block in blockers)
    if throne_adjacent and len(blockers) == 4 and sum(block[1] == Piece.ATTACKER for block in blockers) == 3:
        return True

    # Condition 3: Adjacent to a Camp, opposite side is an Attacker, other two sides are Attackers
    for block in blockers:
        if block[0] in CAMPS:
            camp_pos = block[0]

            # Calculate the opposite position
            if camp_pos[0] == x:  # Camp is along the same row
                opposite_pos = (x, 2 * y - camp_pos[1])
            elif camp_pos[1] == y:  # Camp is along the same column
                opposite_pos = (2 * x - camp_pos[0], y)
            else:
                continue  # Skip if camp is not directly adjacent

            # Validate the opposite position and check the capture condition
            if (0 <= opposite_pos[0] < board.height and
                0 <= opposite_pos[1] < board.width and
                board.get_piece(opposite_pos) == Piece.ATTACKER):
                return True

    # If none of the conditions are met, the King is not captured
    return False
