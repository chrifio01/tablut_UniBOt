from pydantic import BaseModel, ConfigDict
from typing import Annotated, List
from ..consts import INITIAL_STATE
import numpy as np
from .game_utils import *
import json

__all__ = ['State', 'strp_state']

class State(BaseModel):
    """
    Model class representing the states of the game in Tablut.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    board: Annotated[Board, "The current state of the game board"]
    turn: Annotated[Color, "The turn player color"]


def strp_state(state_str: str) -> Annotated[State, "The corresponding state from a string representation of the state sent from the server"]:
    splitted_state_str = state_str.split('-')
    board_state_str = splitted_state_str[0]
    turn_str = splitted_state_str[1][-1] # cause the string starts with \n
    
    pieces = strp_board(board_state_str)
    board = Board(pieces)
    board.pieces = pieces # if it's not the initial state the singleton won't set the passed pieces configuration
    
    return State(board=board, turn=Color(turn_str))