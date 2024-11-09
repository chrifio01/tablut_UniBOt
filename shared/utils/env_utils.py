from pydantic import BaseModel
from typing import Annotated, List
from ..consts import INITIAL_STATE
import numpy as np
from .game_utils import *
import json

class __Action(BaseModel):
    from_: str
    to_: str
    turn: Color
    
    def __str__(self) -> str:
        return json.dumps(
            {
                "from": self.from_,
                "to": self.to_,
                "turn": self.turn.value
            },
            indent=4
        )
class __Board:
    """
    Model class representing the game board in Tablut.
    """
    
    def __init__(
            self, 
            initial_board_state: Annotated[np.ndarray, "The current pieces configuration as a matrix of height x width dim Piece objs"]
        ):
        shape = initial_board_state.shape
        self.__height = shape[0]
        self.__width = shape[1]
        self.__pieces = initial_board_state
        self._initialized = True
    
    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def pieces(self) -> Annotated[List[List[Piece]], "The current pieces configuration as a matrix of height x width dim Piece objs"]:
        return self.__pieces
    
    @pieces.setter
    def pieces(self, new_board_state: Annotated[np.ndarray, "The current pieces configuration as a matrix of height x width dim Piece objs"]) -> None:
        self.__pieces = new_board_state
        
    def update_pieces(self, action: __Action) -> None:
        pass
    
    def __str__(self) -> str:
        return [[p.value for p in self.__pieces[i]].join('') for i in self.__pieces].join('\n')

   
class State(BaseModel):
    """
    Model class representing the states of the game in Tablut.
    """
    board: Annotated[__Board, "The current state of the game board"]
    turn: Annotated[Color, "The turn player color"]


def strp_state(state_str: str) -> Annotated[State, "The corresponding state from a string representation of the state sent from the server"]:
    splitted_state_str = state_str.split('-')
    board_state_str = splitted_state_str[0]
    turn_str = splitted_state_str[1][-1] # cause the string starts with \n
    
    pieces = np.array([list(row) for row in board_state_str.split('\n')[:-1]], dtype=Piece)
    
    return State(board=__Board(pieces), turn=Color(turn_str))