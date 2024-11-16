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

from typing import Annotated
from pydantic import BaseModel, ConfigDict
from shared.consts import WEIGHTS
from .game_utils import Color, Board, strp_board, Piece
import numpy as np


__all__ = ['State', 'strp_state']


class State(BaseModel):
    """
    Model class representing the state of the game in Tablut.

    Attributes:
        board (Board): The current state of the game board, represented as a 2D array of `Piece` values.
        turn (Color): The player whose turn it currently is, represented as a `Color` enum.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    board: Annotated[Board, "The current state of the game board"]
    turn: Annotated[Color, "The turn player color"]


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

        return State(board=board, turn=Color(turn_str))
    except IndexError as e:
        raise ValueError("Invalid state format: missing board or turn information.") from e






############################################### Definition of the functions for the evaluation of the Fitness in the heuristic ###########################################################################



def king_distance_from_center(board: Board, king: tuple [int, int]):
    """
    Calculate de distance of the king from the center

    Args:
    a Board object 
    The king coordinates as a tuple
    """

    return ((king[0] - (board.width//2 + 1))**2 + (king[1] - (board.height//2 + 1))**2)**0.5


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

    if board.get_piece()[king[0]+1][king[1]] == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0]+1, king[1]))

    if board.get_piece()[king[0]-1][king[1]] == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0]-1, king[1]))

    if board.get_piece()[king[0]][king[1]+1] == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0], king[1]+1))

    if board.get_piece()[king[0]][king[1]-1] == Piece.ATTACKER:
        c += 1
        blocked_pos.append((king[0], king[1]-1))
  
    return c, blocked_pos




def position_weight(king: tuple [int, int]):
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
    for i in range(-distance, distance+1):
        for j in range(-distance, distance+1):
            if i == 0 and j == 0:
                continue
            if (x+i, y+j) in board.get_black_coordinates():
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
    state_pieces = {Piece.DEFENDER : 0,
                    Piece.KING : 1,
                    Piece.ATTACKER : 2,
                    Piece.EMPTY : 3,
                    Piece.CAMPS : 4,
                    Piece.THRONE : 4}
    return state_pieces[piece]

class FeaturizeState:
    def __init__(self, state_string: str):
        """
        Initialize the State Featurizer class with the string representing the board
        This string is used to generate the tensor input for the DQN
        
        Arg:
        state_string, the string representing the board
        """
        self.state_string = state_string

    def generate_input(self):
        """
        Return the tensor representing the state which the DQN should receive as input to choose best action

        """
        position_layer = [np.zeros((9, 9), dtype=bool) for _ in range(5)]
        position_layer[piece_parser(Piece.CAMPS)] = np.array([
                                                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                                            [1, 1, 0, 0, 1, 0, 0, 1, 1],  
                                                            [1, 0, 0, 0, 0, 0, 0, 0, 1],  
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],  
                                                            [0, 0, 0, 0, 1, 0, 0, 0, 0],  
                                                            [0, 0, 0, 1, 1, 1, 0, 0, 0]   
                                                            ], dtype=bool)
        state = strp_state(self.state_string)
        board_str = Board(state.board.pieces)
        
        for i in range(board_str.height):
            for j in range(board_str.width):
                index = i * 9 + j
                position = (i,j)  
                piece = piece_parser(Piece(board_str.get_piece(position)))
                position_layer[piece][i, j] = True
        
        turn_layer = np.array([1 if Color(state.turn) == 'W' else 0], dtype=bool)

        w_heur_layer = np.array([board_str.num_black(), board_str.num_white(), king_distance_from_center(board_str,board_str.king_pos()), position_weight(board_str.king_pos())])
        #king_surrounded(board_str)[0],
        b_heur_layer = np.array([board_str.num_black(), board_str.num_white(), pawns_around(board_str, board_str.king_pos(), 1)])

        input_tensor = {"board_input": position_layer,
                        "turn_input": turn_layer, 
                        "white_input": w_heur_layer,
                        "black_input": b_heur_layer}
        return input_tensor
