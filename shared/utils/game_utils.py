"""
This module defines core components for the Tablut game, including `Color`, `Piece`, `Board`, and `Action` classes,
along with utility functions for parsing and formatting board and position strings.

Classes:
    Color: Enum representing player colors (WHITE and BLACK).
    Piece: Enum for different game pieces, including DEFENDER, ATTACKER, KING, THRONE, and EMPTY.
    Action: Model for representing a player's action, including the starting position, destination, and turn.
    Board: Singleton class representing the Tablut board, managing piece positions and board properties.
    Turn: Enum representing the possible states of a player's turn in the Tablut game.

Functions:
    strp_board(board_str: str) -> np.ndarray:
        Parses a board string from the server and converts it into an `np.ndarray` of `Piece` values.
        
    strp_square(square_str: str) -> Tuple[int, int]:
        Parses a string representation of a square (e.g., "a1") to board coordinates (row, column).
        
    strf_square(position: Tuple[int, int]) -> str:
        Formats a board coordinate (row, column) back into a string representation (e.g., "a1").

    strp_color(color_str: str) -> Color:
        Parses a color string (e.g., "W" or "B") and returns the corresponding `Color` enum value.

    strp_turn(turn_str: str) -> Turn:
        Parses a turn string (e.g., "W", "B", "WB", "BW" or "D") and returns the corresponding `Turn` enum value.

    parse_state_board(state_board: List[List[str]]) -> Board:
        Parses a 2D list of piece strings into a `Board` object.

Usage Example:
    Initialize and update the board state:
        initial_state_str = "OOOBBBOOO\nOOOOBOOOO\n... - WHITE"
        state = strp_state(initial_state_str)
        action = Action(from_="d5", to_="e5", turn=Color.WHITE)
        state.board.update_pieces(action)
"""

from enum import Enum
import string
import json
from typing import Annotated, Tuple, List, Union
from pydantic import BaseModel
import numpy as np

from shared.consts import CAMPS, WIN_TILES

__all__ = ['Color', 'Piece', 'Board', 'Action', 'strp_board', 'strf_square', 'strp_square', 'strp_turn', 'Turn',
           'strp_color']


class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.

    Attributes:
        WHITE: Represents the white pieces or defenders.
        BLACK: Represents the black pieces or attackers.
    """
    WHITE = 'W'
    BLACK = 'B'


def strp_color(color_str: str) -> Color:
    """
    Parses a turn string (e.g., "W", "B", "WB", "BW" or "D") and returns the corresponding `Turn` enum value.

    Args:
        turn_str (str): The turn string to parse.

    Returns:
        Turn: The corresponding `Turn` enum value.
    """
    low = color_str.lower()

    if low == 'white':
        return Color.WHITE
    if low == 'black':
        return Color.BLACK
    raise ValueError(f"Invalid color string: {color_str}")


class Turn(Enum):
    """
    Enum representing the possible states of a player's turn in the Tablut game.

    Attributes:
        BLACK_TURN: Indicates that it is the black player's turn.
        WHITE_TURN: Indicates that it is the white player's turn.
        BLACK_WIN: Indicates that the black player has won the game.
        WHITE_WIN: Indicates that the white player has won the game.
        DRAW: Indicates that the game has ended in a draw.
    """
    BLACK_TURN = 'B'
    WHITE_TURN = 'W'
    BLACK_WIN = 'BW'
    WHITE_WIN = 'WW'
    DRAW = 'D'


def strp_turn(turn_str: str) -> Turn:
    """
    Parses a turn string (e.g., "W", "B", "WB", "BW" or "D") and returns the corresponding `Turn` enum value.

    Args:
        turn_str (str): The turn string to parse.

    Returns:
        Turn: The corresponding `Turn` enum value.
    """
    low = turn_str.lower()

    if low == 'white':
        return Turn.WHITE_TURN
    if low == 'black':
        return Turn.BLACK_TURN
    if low == 'whitewin':
        return Turn.WHITE_WIN
    if low == 'blackwin':
        return Turn.BLACK_WIN
    if low == 'draw':
        return Turn.DRAW
    raise ValueError(f"Invalid turn string: {turn_str}")

def winner_color(turn: Turn) -> Union[Color, None]:
    """
    Determines the winner's color based on the turn state.

    Args:
        turn (Turn): The current turn state.

    Returns:
        Union[Color, None]: The color of the winning player if there is a winner, otherwise None.

    Raises:
        ValueError: If the turn state indicates that the game is still ongoing.
    """
    if turn == Turn.WHITE_WIN:
        return Color.WHITE
    if turn == Turn.BLACK_WIN:
        return Color.BLACK
    if turn in (Turn.WHITE_TURN, Turn.BLACK_TURN):
        raise ValueError("No Winner")
    return None

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
    turn: Turn

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
    CAMPS = 'C'


def _strp_piece(piece_str: str) -> Piece:
    lower_str = piece_str.lower()
    if lower_str == 'empty':
        return Piece.EMPTY
    if lower_str == 'white':
        return Piece.DEFENDER
    if lower_str == 'black':
        return Piece.ATTACKER
    if lower_str == 'king':
        return Piece.KING
    if lower_str == 'throne':
        return Piece.THRONE
    raise ValueError(f"Invalid piece string: {piece_str}")


def strp_board(board_str: str) -> Annotated[
    np.ndarray, "The corresponding board configuration from a string representation of the pieces sent from the server"]:
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


def _check_single_king_and_throne(pieces: np.ndarray) -> bool:
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

    if center not in (Piece.THRONE, Piece.KING) and throne_count == 0:
        raise ValueError("Invalid board: center has no THRONE or KING.")
    if center not in (Piece.THRONE,) and throne_count == 1:
        raise ValueError("Invalid board: the THRONE is not in the center")

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

    def __init__(
            self,
            initial_board_state: Annotated[
                np.ndarray, "The initial pieces configuration as a 2D np array referenced as (col, row) pairs"]
    ):
        """
        Initializes the board with an initial state and validates it.

        Args:
            initial_board_state (np.ndarray): The board's initial configuration as a 2D array of Piece values.
        
        Raises:
            ValueError: If there are multiple KINGs or THRONEs on the board.
        """
        _check_single_king_and_throne(initial_board_state)

        shape = initial_board_state.shape
        self.__height = shape[0]  # first index is the row
        self.__width = shape[1]  # second index is the column
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
    def pieces(self) -> Annotated[
        np.ndarray, "The current pieces configuration as a matrix of height x width dim Piece objs"]:
        """Returns the current board configuration."""
        return self.__pieces

    @pieces.setter
    def pieces(self, new_board_state: Annotated[
        np.ndarray, "The new pieces configuration sent from the server converted in np.array"]) -> None:
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

        _check_single_king_and_throne(new_board_state)

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
        if action.turn.value == Color.WHITE.value and moving_piece not in (Piece.DEFENDER, Piece.KING):
            raise ValueError("Cannot move opponent's pieces.")
        if action.turn.value == Color.BLACK.value and moving_piece != Piece.ATTACKER:
            raise ValueError("Cannot move opponent's pieces.")
        if from_indexes == (self.__height // 2, self.__width // 2) and moving_piece == Piece.KING:
            self.__pieces[from_indexes] = Piece.THRONE
        else:
            self.__pieces[from_indexes] = Piece.EMPTY
        self.__pieces[to_indexes] = moving_piece
        
        # Check for captures around the destination
        adjacent_positions = [
            (to_indexes[0] + 1, to_indexes[1]),
            (to_indexes[0] - 1, to_indexes[1]),
            (to_indexes[0], to_indexes[1] + 1),
            (to_indexes[0], to_indexes[1] - 1),
        ]

        for pos in adjacent_positions:
            if 0 <= pos[0] < self.__height and 0 <= pos[1] < self.__width:
                captured_piece = self.get_piece(pos)

                # Only check for captures of opponent's pieces
                if action.turn == Turn.BLACK_TURN and captured_piece == Piece.DEFENDER:
                    if self._is_a_capture(pos, captured_piece):
                        self.__pieces[pos] = Piece.EMPTY
                elif action.turn == Turn.WHITE_TURN and captured_piece == Piece.ATTACKER:
                    if self._is_a_capture(pos, captured_piece):
                        self.__pieces[pos] = Piece.EMPTY

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
    
    def _is_a_capture(self, moving_piece_coords: Tuple[int, int], moving_piece_type: Piece) -> bool:
        """
        Checks if a piece at a given position is captured.

        Args:
            moving_piece_coords (Tuple[int, int]): The (row, column) position of the moving piece.
            moving_piece_type (Piece): The type of the moving piece (ATTACKER, DEFENDER, or KING).

        Returns:
            bool: True if the piece is captured, False otherwise.
        """
        assert moving_piece_type in (Piece.ATTACKER, Piece.DEFENDER, Piece.KING)
        
        x, y = moving_piece_coords
        adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        
        def is_enemy_or_threat(pos: Tuple[int, int]) -> bool:
            """Helper to determine if a position contains an enemy, throne, or camp."""
            row, col = pos
            if not (0 <= row < self.height and 0 <= col < self.width):
                return False  # Out of bounds
            piece = self.get_piece(pos)
            if moving_piece_type == Piece.DEFENDER or moving_piece_type == Piece.KING:
                return piece == Piece.ATTACKER or pos in CAMPS or piece == Piece.THRONE
            elif moving_piece_type == Piece.ATTACKER:
                return piece in {Piece.DEFENDER, Piece.KING}
            return False

        # Check surrounding positions to determine capture
        threats_count = sum(is_enemy_or_threat(pos) for pos in adjacent_positions)

        # Regular rules for Attackers and Defenders:
        return threats_count >= 2
    
    def num_threats_to_piece(self, piece: tuple, piece_type: Piece) -> int:
        """
        Calculates the number of threats to a piece at a given position.

        Args:
            piece (tuple): Coordinates (row, column) of the piece to evaluate.
            piece_type (Piece): Type of the piece (e.g., Piece.KING, Piece.DEFENDER).

        Returns:
            int: Number of threats (attackers or strategic blockers) adjacent to the piece.
        """
        x, y = piece
        threats = 0
        adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for pos in adjacent_positions:
            if 0 <= pos[0] < self.__height and 0 <= pos[1] < self.__width:
                threat_piece = self.get_piece(pos)
                if piece_type == Piece.KING:
                    if threat_piece in {Piece.ATTACKER, Piece.THRONE} or pos in CAMPS:
                        threats += 1
                elif piece_type == Piece.DEFENDER and threat_piece == Piece.ATTACKER:
                    threats += 1
        return threats
    
    def king_free_escape_routes(self, king_pos: tuple) -> int:
        """
        Evaluates the number of clear escape routes for the king.

        Args:
            king_pos (tuple): Coordinates (row, column) of the king.

        Returns:
            int: Number of clear escape routes for the king.
        """
        escape_routes = 0
        escape_positions = [
            (king_pos[0], self.width - 1),  # Right edge
            (king_pos[0], 0),              # Left edge
            (self.height - 1, king_pos[1]),  # Bottom edge
            (0, king_pos[1])               # Top edge
        ]
        for pos in escape_positions:
            if self.is_there_a_clear_view(king_pos, pos):
                escape_routes += 1
        return escape_routes
    
    def king_pos(self):
        """
        Return the king position on the board as a tuple of two elements.
        Raises a ValueError if no king is found.
        """
        king_position = np.where(self.__pieces == Piece.KING)

        if king_position[0].size == 0:
            raise ValueError("King not found on the board")

        return (king_position[0][0], king_position[1][0])

    def num_white(self):
        """
        Return the number of white pawns on the board
        """
        return np.count_nonzero(self.__pieces == Piece.DEFENDER)

    def num_black(self):
        """
        Return the number of black pawns on the board
        """
        return np.count_nonzero(self.__pieces == Piece.ATTACKER)

    def is_there_a_clear_view(self, piece1: tuple, piece2: tuple):
        """"
        Checks if there is a clear line of sight between two pieces on a grid (same row or column).
        It returns True if the pieces are aligned horizontally or vertically and there are no other
        pieces between them. If the pieces are not aligned or there are obstacles
        in the line of sight, it returns False.

        Arg:
        two tuple with the coordinates of the pieces
        """

        if piece1[0] == piece2[0]:
            offset = 1 if piece1[1] <= piece2[1] else -1
            for i in range(piece1[1] + offset, piece2[1], offset):
                if self.__pieces[piece1[0]][i] != Piece.EMPTY:
                    return False
            return True
        if piece1[1] == piece2[1]:
            offset = 1 if piece1[0] <= piece2[0] else -1
            for i in range(int(piece1[0] + offset), int(piece2[0]), offset):
                if self.__pieces[i][piece1[1]] != Piece.EMPTY:
                    return False
            return True

        return False

    def get_black_coordinates(self):
        """
        Function that return a list of all the coordinates for the black pawns on the board at the moment
        """
        return [(i, j) for i in range(self.__pieces.shape[0]) for j in range(self.__pieces.shape[1]) if
                self.__pieces[i, j] == Piece.ATTACKER]

    def is_tile_free(self, tile):
        # Check if the specified tile is free (not occupied by any piece)
        row, col = tile
        return self.__pieces[row][col] == Piece.EMPTY
    
    def king_proximity_to_escape(self):
        """
        Calculate the minimum Manhattan distance from the king to the escape (WIN) tiles.
        
        Returns:
            int: The minimum Manhattan distance to the closest escape tile, or float('inf') if no escape tiles are reachable.
        """
        king_position = self.king_pos()
        min_distance = float('inf')  # Start with an infinitely large distance
        
        # Iterate over each WIN TILE to calculate proximity
        for tile in WIN_TILES:
            if self.is_tile_free(tile):  # Only consider free WIN TILES
                manhattan_distance = abs(king_position[0] - tile[0]) + abs(king_position[1] - tile[1])
                
                # Update minimum distance if the current one is smaller
                if manhattan_distance < min_distance:
                    min_distance = manhattan_distance

        # Return the minimum distance found or float('inf') if no valid tiles were found
        return min_distance if min_distance != float('inf') else None

def parse_state_board(state_board: List[List[str]]) -> Board:
    """
    Parses a 2D list of piece strings into a Board object.

    Args:
        state_board (List[List[str]]): A 2D list where each element is a string representing a piece.

    Returns:
        Board: A Board object initialized with the parsed pieces.
    """
    pieces = np.array([[_strp_piece(piece_str) for piece_str in row] for row in state_board])
    return Board(pieces)
