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
from .game_utils import Board, strp_board, Piece, strp_turn, parse_state_board, Turn

__all__ = ['State', 'strp_state', 'state_decoder']


class State(BaseModel):
    """
    Model class representing the state of the game in Tablut.

    Attributes:
        board (Board): The current state of the game board, represented as a 2D array of `Piece` values.
        turn (Color): The player whose turn it currently is, represented as a `Color` enum.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    board: Annotated[Board, "The current state of the game board"]
    turn: Annotated[Turn, "The turn player"]
    
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
