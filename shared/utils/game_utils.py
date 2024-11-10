"""
This module defines core components for the Tablut game, including `Color`, `Piece`, `Board`, and `Action` classes,
along with utility functions for parsing and formatting board and position strings.

Classes:
    Color: Enum representing player colors (WHITE and BLACK).
    Piece: Enum for different game pieces, including DEFENDER, ATTACKER, KING, THRONE, and EMPTY.
    Action: Model for representing a player's action, including the starting position, destination, and turn.
    Board: Singleton class representing the Tablut board, managing piece positions and board properties.

Functions:
    strp_board(board_str: str) -> np.ndarray:
        Parses a board string from the server and converts it into an `np.ndarray` of `Piece` values.
        
    strp_square(square_str: str) -> Tuple[int, int]:
        Parses a string representation of a square (e.g., "a1") to board coordinates (row, column).
        
    strf_square(position: Tuple[int, int]) -> str:
        Formats a board coordinate (row, column) back into a string representation (e.g., "a1").

Usage Example:
    Initialize and update the board state:
        initial_state_str = "OOOBBBOOO\nOOOOBOOOO\n... - WHITE"
        state = strp_state(initial_state_str)
        action = Action(from_="d5", to_="e5", turn=Color.WHITE)
        state.board.update_pieces(action)
"""

from enum import Enum
from typing import Annotated, Tuple
import json
import string
from pydantic import BaseModel
import numpy as np

__all__ = ['Color', 'Piece', 'Board', 'Action', 'strp_board', 'strf_square', 'strp_square']

class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.

    Attributes:
        WHITE: Represents the white pieces or defenders.
        BLACK: Represents the black pieces or attackers.
    """
    WHITE = 'W'
    BLACK = 'B'

class Action(BaseModel):
    """
    Model representing a player's move, consisting of the start and destination squares and the player's color.

    Attributes:
        from_ (str): The starting square of the move, in chess notation (e.g., "d5").
        to_ (str): The destination square of the move, in chess notation.
        turn (Color): The color of the player making the move.

    Methods:
        __str__: Returns a JSON string representation of the action.
    """
    from_: str
    to_: str
    turn: Color
    
    def __str__(self) -> str:
        """
        Returns a JSON-formatted string representing the action.

        Returns:
            str: JSON string with "from", "to", and "turn" attributes.
        """
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

    Attributes:
        DEFENDER: The defender piece (white).
        ATTACKER: The attacker piece (black).
        KING: The king piece, belonging to the white player.
        THRONE: The central throne position on the board.
        EMPTY: An empty cell on the board.
    """
    DEFENDER = 'W'
    ATTACKER = 'B'
    KING = 'K'
    THRONE = 'T'
    EMPTY = 'O'

def strp_board(board_str: str) -> Annotated[np.ndarray, "The corresponding board configuration from a string representation of the pieces sent from the server"]:
    """
    Converts a board string representation into a numpy array of `Piece` values.

    Args:
        board_str (str): A string representation of the board, with rows separated by newline characters.
    
    Returns:
        np.ndarray: A 2D array with `Piece` values representing the board state.
    """
    rows = board_str.strip().split('\n')
    board_array = np.array([[Piece(char) for char in row] for row in rows[::-1]], dtype=Piece)
    return board_array

def strp_square(square_str: str) -> Tuple[int, int]:
    """
    Parses a square in chess notation to a row, column tuple.

    Args:
        square_str (str): The square in chess notation (e.g., "a1").
    
    Returns:
        Tuple[int, int]: The (row, column) position on the board.
    
    Raises:
        ValueError: If `square_str` is not valid chess notation.
    """
    if len(square_str) != 2:
        raise ValueError("Invalid square format")
    
    if square_str[0].lower() not in string.ascii_lowercase or square_str[1] not in string.digits:
        raise ValueError("Invalid square format")
    
    column = ord(square_str[0].lower()) - ord('a')
    row = int(square_str[1]) - 1
    
    return row, column

def strf_square(position: Tuple[int, int]) -> str:
    """
    Converts a (row, column) position to chess notation.

    Args:
        position (Tuple[int, int]): The position on the board.
    
    Returns:
        str: Chess notation string for the position.
    
    Raises:
        ValueError: If `position` is out of bounds.
    """
    if position[1] > len(string.ascii_lowercase) - 1 or position[0] < 0:
        raise ValueError("Invalid position")
    
    column = string.ascii_lowercase[position[1]]
    row = position[0] + 1
    
    return f"{column}{row}"

def __check_single_king_and_throne(pieces: np.ndarray) -> bool:
    """
    Validates that there is exactly one KING and one THRONE on the board.

    Args:
        pieces (np.ndarray): Board configuration to validate.
    
    Returns:
        bool: True if the board is valid.
    
    Raises:
        ValueError: If multiple KINGs, multiple THRONEs, or misplaced THRONE.
    """
    king_count = np.count_nonzero(pieces == Piece.KING)
    throne_count = np.count_nonzero(pieces == Piece.THRONE)
    
    if king_count > 1:
        raise ValueError("Invalid board: more than one KING found.")
    if king_count == 0:
        raise ValueError("Invalid board: no KING found.")
    
    if throne_count > 1:
        raise ValueError("Invalid board: more than one THRONE found.")
    
    center = pieces[pieces.shape[0] // 2][pieces.shape[1] // 2]
    
    if center not in (Piece.THRONE, Piece.KING):
        raise ValueError("Invalid board: THRONE not in the center.")
    
    return True

class Board:
    """
    Singleton class representing the game board in Tablut.

    Attributes:
        height (int): The height of the board.
        width (int): The width of the board.
        pieces (np.ndarray): The current configuration of the board.

    Methods:
        update_pieces(action: Action): Updates board state based on an action.
        get_piece(position: Tuple[int, int]) -> Piece: Returns the piece at a specific position.
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
        """
        Initializes the board with an initial state and validates it.

        Args:
            initial_board_state (np.ndarray): The board's initial configuration as a 2D array of Piece values.
        
        Raises:
            ValueError: If there are multiple KINGs or THRONEs on the board.
        """
        __check_single_king_and_throne(initial_board_state)
        
        if not hasattr(self, '_initialized'):
            shape = initial_board_state.shape
            self.__height = shape[0]    # first index is the row
            self.__width = shape[1]     # second index is the column
            self.__pieces = initial_board_state
            self._initialized = True
    
    @property
    def height(self) -> int:
        """Returns the board height."""
        return self.__height
    
    @property
    def width(self) -> int:
        """Returns the board width."""
        return self.__width
    
    @property
    def pieces(self) -> Annotated[np.ndarray, "The current pieces configuration as a matrix of height x width dim Piece objs"]:
        """Returns the current board configuration."""
        return self.__pieces
    
    @pieces.setter
    def pieces(self, new_board_state: Annotated[np.ndarray, "The new pieces configuration sent from the server converted in np.array"]) -> None:
        """
        Updates the board configuration, ensuring valid dimensions and piece constraints.

        Args:
            new_board_state (np.ndarray): The new configuration for the board.
        
        Raises:
            ValueError: If `new_board_state` has incompatible dimensions or multiple KINGs/THRONEs.
        """
        shape = new_board_state.shape
        if shape[0] > self.__height or shape[1] > self.__width:
            raise ValueError("Invalid new board state size")
        
        __check_single_king_and_throne(new_board_state)
        
        self.__pieces = new_board_state
        
    def update_pieces(self, action: Action) -> None:
        """
        Executes an action by moving a piece from start to destination on the board.

        Args:
            action (Action): The action to apply on the board.
        
        Raises:
            ValueError: If the piece cannot legally move.
        """
        from_indexes = strp_square(action.from_)
        to_indexes = strp_square(action.to_)
        
        moving_piece = self.__pieces[from_indexes]
        
        if moving_piece not in (Piece.DEFENDER, Piece.ATTACKER, Piece.KING):
            raise ValueError(f"Cannot move {moving_piece} from {action.from_} to {action.to_}.")
        if from_indexes == (4,4) and moving_piece == Piece.KING:
            self.__pieces[from_indexes] = Piece.THRONE
        else:
            self.__pieces[from_indexes] = Piece.EMPTY
        self.__pieces[to_indexes] = moving_piece
        
    def get_piece(self, position: Tuple[int, int]) -> Piece:
        """
        Returns the piece at a given position on the board.

        Args:
            position (Tuple[int, int]): The (row, column) position.
        
        Returns:
            Piece: The piece located at `position`.
        """
        return self.__pieces[position]
    
    def __str__(self) -> str:
        """
        Returns a string representation of the board's current state.

        Returns:
            str: A string representation of the board.
        """
        return '\n'.join(''.join(piece.value for piece in row) for row in self.__pieces[::-1])
