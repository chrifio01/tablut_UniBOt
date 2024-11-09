from pydantic import BaseModel
from typing import Annotated, List
from ..consts import INITIAL_STATE
import numpy as np
from .game_utils import *
import json

class State(BaseModel):
    """
    Model class representing the states of the game in Tablut.
    """
    board: Annotated[Board, "The current state of the game board"]
    turn: Annotated[Color, "The turn player color"]


def strp_state(state_str: str) -> Annotated[State, "The corresponding state from a string representation of the state sent from the server"]:
    splitted_state_str = state_str.split('-')
    board_state_str = splitted_state_str[0]
    turn_str = splitted_state_str[1][-1] # cause the string starts with \n
    
    pieces = np.array([list(row) for row in board_state_str.split('\n')[:-1]], dtype=Piece)
    
    return State(board=Board(pieces), turn=Color(turn_str))