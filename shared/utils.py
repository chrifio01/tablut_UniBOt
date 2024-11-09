from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Annotated, List
from .consts import INITIAL_BOARD_STATE
from enum import Enum

class Piece(Enum):
    """
    Enum representing the pieces in Tablut.
    """
    DEFENDER = 'W'
    ATTACKER = 'B'
    KING = 'K'
    THRONE = 'T'
    EMPTY = 'O'

def strp_pieces(board_str: str) -> Annotated[List[List[Piece]], "The corresponding pieces configuration from a string representation of the board"]:
    pieces = []
    for row in board_str.split('\n'):
        pieces.append([Piece(ch) for ch in row])
    return pieces

class Board:
    """
    Model class representing the game board in Tablut.
    """
    def __init__(self, initial_board_state: str = INITIAL_BOARD_STATE, height: int = 9, width: int = 9):
        self.__height = height
        self.__width = width
        self.__pieces = strp_pieces(initial_board_state)
    
    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def pieces(self) -> Annotated[List[List[Piece]], "The current pieces configuration as a matrix of height x width dim"]:
        return self.__pieces
    
    @pieces.setter
    def pieces(self, new_board_state: str) -> None:
        self.__pieces = strp_pieces(new_board_state)
        
    def update_pieces(self, action) -> None:
        pass
    
    def __str__(self) -> str:
        return [[p.value for p in self.__pieces[i]].join('') for i in self.__pieces].join('\n')

class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.
    """
    WHITE ='W'
    BLACK ='B'
    
class State(BaseModel):
    """
    Model class representing the states of the game in Tablut.
    """
    board: Annotated[Board, "The current state of the game board"]
    turn: Annotated[Color, "The turn player color"]
    

class AbstractPlayer(ABC):
    """
    Abstract base class for players in Tablut.
    """
    
    @property
    @abstractmethod
    def current_state(self):
        pass
    
    @current_state.setter
    @abstractmethod
    def current_state(self, new_state):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def color(self) -> Color:
        pass
    
    @abstractmethod
    def send_move(self):
        pass
    
    @abstractmethod
    def fit(self, state, *args, **kwargs):
        pass