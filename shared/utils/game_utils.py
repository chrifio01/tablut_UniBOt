from enum import Enum
from typing import Annotated, List
import numpy as np

__all__ = ['Color', 'Piece', 'Board']

class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.
    """
    WHITE ='W'
    BLACK ='B'
    
class Piece(Enum):
    """
    Enum representing the pieces in Tablut.
    """
    DEFENDER = 'W'
    ATTACKER = 'B'
    KING = 'K'
    THRONE = 'T'
    EMPTY = 'O'
    
class Board:
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
    
    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def pieces(self) -> Annotated[List[List[Piece]], "The current pieces configuration as a matrix of height x width dim Piece objs"]:
        return self.__pieces
        
    def update_pieces(self, action: __Action) -> None:
        pass
    
    def __str__(self) -> str:
        return [[p.value for p in self.__pieces[i]].join('') for i in self.__pieces].join('\n')