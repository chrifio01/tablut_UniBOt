from pydantic import BaseModel
from typing import Annotated, List
from ..consts import INITIAL_BOARD_STATE
import numpy as np

class __Board:
    """
    Model class representing the game board in Tablut.
    """
    _instance = None  # Class-level attribute to store the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(__Board, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(
            self, 
            initial_board_state: Annotated[np.ndarray, "The current pieces configuration as a matrix of height x width dim Piece objs"]
        ):
        if not hasattr(self, '_initialized'):  # Ensures `__init__` runs only once
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
        
    def update_pieces(self, action: Action) -> None:
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
    pieces = []
    for row in board_str.split('\n'):
        pieces.append([Piece(ch) for ch in row])
    return pieces

class Action(BaseModel):
    from_: str
    to_: str
    turn: Color