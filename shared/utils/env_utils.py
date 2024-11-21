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
from shared.consts import WEIGHTS, CAMPS, INITIAL_STATE
from .game_utils import Board, strp_board, Piece, strp_turn, parse_state_board, Turn, Color

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
    elif board.get_piece((king[0] + 1, king[1])) == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0] + 1, king[1]))
    if king[0] - 1 < 0:
        c += 1
    elif board.get_piece((king[0] - 1, king[1])) == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0] - 1, king[1]))
    if king[1] + 1 >= board.width:
        c += 1
    elif board.get_piece((king[0], king[1] + 1)) == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0], king[1] + 1))
    if king[1] - 1 < 0:
        c += 1
    elif board.get_piece((king[0], king[1] - 1)) == Piece.ATTACKER:
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

class FeaturizedState(BaseModel):
    
    board_input: np.ndarray
    turn_input: np.ndarray
    white_input: np.ndarray
    black_input: np.ndarray
    
    class Config:
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
                    position_layer[piece][-i - 1, j] = True
                except KeyError:
                    pass

        turn_layer = np.array([1 if player_color == Color.WHITE else 0], dtype=bool)
        
        w_heur_layer = np.array(
            [board.num_black(), board.num_white(), king_distance_from_center(board, board.king_pos()),
             king_surrounded(board)[0], position_weight(board.king_pos())])

        b_heur_layer = np.array(
            [board.num_black(), board.num_white(), pawns_around(board, board.king_pos(), 1)])

        return FeaturizedState(board_input=position_layer, turn_input=turn_layer, white_input=w_heur_layer, black_input=b_heur_layer)


def black_win_con(board: Board, king: tuple[int, int]):
    """
    Black player win condition is satisfied when the value of this function is 4, the king is surrounded on every side

    Arg:
    Board object
    The king postition as a tuple of int

    Return: 
    The number of blocked sides of the king 
    """
    x, y = king
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (x + i, y + j) in board.get_black_coordinates() or board.get_piece(
                    (x + i, y + j)) == Piece.THRONE or board.get_piece((x + i, y + j)) == Piece.CAMPS:
                count += 1
    return count
