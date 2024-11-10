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
    rows = board_str.strip().split('\n')
    board_array = np.array([[Piece(char) for char in row] for row in rows[::-1]], dtype=Piece)
    return board_array

def strp_square(square_str: str) -> Tuple[int, int]:
    if square_str[0].lower() not in string.ascii_lowercase or square_str[1] not in string.digits:
        raise ValueError("Invalid square format")
    
    column = ord(square_str[0].lower()) - ord('a')
    row = int(square_str[1]) - 1
    
    return row, column

def strf_square(position: Tuple[int, int]) -> str:
    if position[1] > len(string.ascii_lowercase) - 1 or position[0] < 0:
        raise ValueError("Invalid position")
    
    column = string.ascii_lowercase[position[1]]
    row = position[0] + 1
    
    return f"{column}{row}"

class Board:
    """
    Model class representing the game board in Tablut.
    """
    _instance = None  # Class-level attribute to store the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Board, cls).__new__(cls)
        return cls._instance
    
    def __init__(
            self, 
            initial_board_state: Annotated[np.ndarray, "The initial pieces configuration as a 2D np array referenced as (col, row) pairs"]
        ):
        self.__class__.__check_single_king_and_throne(initial_board_state)
        
        if not hasattr(self, '_initialized'):
            shape = initial_board_state.shape
            self.__height = shape[0]    # first index is the row
            self.__width = shape[1]     # second index is the column
            self.__pieces = initial_board_state
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
    def pieces(self, new_board_state: Annotated[np.ndarray, "The new pieces configuration sent from the server converted in np.array"]) -> None:
        shape = new_board_state.shape
        if shape[0] > self.__height or shape[1] > self.__width:
            raise ValueError("Invalid new board state size")
        
        self.__class__.__check_single_king_and_throne(new_board_state)
        
        self.__pieces = new_board_state
        
    def update_pieces(self, action: _Action) -> None:
        from_indexes = strp_square(action.from_)
        to_indexes = strp_square(action.to_)
        
        moving_piece = self.__pieces[from_indexes]
        
        if moving_piece not in (Piece.DEFENDER, Piece.ATTACKER, Piece.KING):
            raise ValueError(f"Cannot move {moving_piece} from {action.from_} to {action.to_}.")
            
        self.__pieces[from_indexes] = Piece.EMPTY
        self.__pieces[to_indexes] = moving_piece
        
    def get_piece(self, position: Tuple[int, int]) -> Piece:
        return self.__pieces[position]
    
    @staticmethod
    def __check_single_king_and_throne(pieces: np.ndarray) -> bool:
        # Count occurrences of KING and THRONE
        king_count = np.count_nonzero(pieces == Piece.KING)
        throne_count = np.count_nonzero(pieces == Piece.THRONE)
        
        # Ensure only one KING and one THRONE on the board
        if king_count > 1:
            raise ValueError("Invalid board: more than one KING found.")
        elif king_count == 0:
            raise ValueError("Invalid board: no KING found.")
        
        if throne_count > 1:
            raise ValueError("Invalid board: more than one THRONE found.")
        
        center = pieces[pieces.shape[0] // 2][pieces.shape[1] // 2]
        
        if center != Piece.THRONE and center != Piece.KING:
            raise ValueError("Invalid board: THRONE not in the center.")
        
        return True
    
    def __str__(self) -> str:
        return '\n'.join(''.join(piece.value for piece in row) for row in self.__pieces[::-1])