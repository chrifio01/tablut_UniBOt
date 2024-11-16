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
from .game_utils import Board, strp_board, strp_turn, parse_state_board, Turn
from shared.loggers import logger

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
    