from enum import Enum
from typing import Annotated, List, Tuple
from pydantic import BaseModel
import numpy as np
import json
import string

__all__ = ['Color', 'Piece', 'Board', '_Action', 'strp_board', 'strf_square', 'strp_square']

class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.
    """
    WHITE ='W'
    BLACK ='B'

class _Action(BaseModel):
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
class Piece(Enum):
    """
    Enum representing the pieces in Tablut.
    """
    DEFENDER = 'W'
    ATTACKER = 'B'
    KING = 'K'
    THRONE = 'T'
    EMPTY = 'O'

def strp_board(board_str: str) -> Annotated[np.ndarray, "The corresponding board configuration from a string representation of the pieces sent from the server"]:    
    return np.array([list(row) for row in board_str.split('\n')[:-1]], dtype=Piece)

def strp_square(square_str: str) -> Tuple[int, int]:
    column= list(range(ord('a'), ord('z') + 1)).index(square_str[0].lower())
    row = int(square_str[1]) - 1  # adjusting to 0-based indexing
    return column, row

def strf_square(position: Tuple[int, int]) -> str:
    column = string.ascii_lowercase[position[0]]
    row = position[1] + 1
    return f"{column}{row}"

class Board:
    """
    Model class representing the game board in Tablut.
    """
    _instance = None  # Class-level attribute to store the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Board, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(
            self, 
            initial_board_state: Annotated[str, "The initial pieces configuration"]
        ):
        if not hasattr(self, '_initialized'):
            shape = initial_board_state.shape
            self.__height = shape[1]    # first index is the column
            self.__width = shape[0]     # second index is the row
            self.__pieces = strp_board(initial_board_state)
            self._initialized = True
    
    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def pieces(self) -> Annotated[np.ndarray, "The current pieces configuration as a matrix of height x width dim Piece objs"]:
        return self.__pieces
    
    @pieces.setter
    def pieces(self, new_board_state: Annotated[str, "The new pieces configuration sent from the server"]) -> None:
        self.__pieces = strp_board(new_board_state)
        
    def update_pieces(self, action: _Action) -> None:
        from_indexes = strp_square(action.from_)
        to_indexes = strp_square(action.to_)
        moving_piece = self.__pieces[from_indexes]
        self.__pieces[from_indexes] = Piece.EMPTY
        self.__pieces[to_indexes] = moving_piece
        
    def get_piece(self, position: Tuple[int, int]) -> Piece:
        return self.__pieces[position]
    
    def __str__(self) -> str:
        return [[p.value for p in self.__pieces[i]].join('') for i in self.__pieces].join('\n')