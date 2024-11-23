"""
This module contains the `MoveChecker` class, which provides methods to validate moves 
and generate possible actions in the game of Tablut.

Classes:
    MoveChecker: Contains methods for move validation and generation of possible actions.
"""

from typing import List, Tuple
import numpy as np
from .utils import State, Action, strf_square, strp_square, Piece, Turn
from .consts import CAMPS
from .exceptions import InvalidAction

class MoveChecker:
    """
    A class that provides methods to check the validity of moves in the Tablut game,
    ensuring they adhere to the game's rules and constraints.
    """

    @staticmethod
    def _is_camp(row: int, col: int) -> bool:
        """
        Determines if a specified position is a camp.

        Args:
            row (int): The row index of the position.
            col (int): The column index of the position.

        Returns:
            bool: True if the position is a camp, False otherwise.
        """
        return (row, col) in CAMPS

    @classmethod
    def _check_for_jumps(
            cls,
            state: State,
            action_from: Tuple[int, int],
            action_to: Tuple[int, int]
    ) -> None:
        """
        Checks for illegal jumps over non-empty spaces, the throne, or camps during a move.

        Args:
            state (State): The current game state.
            action_from (Tuple[int, int]): The starting (row, column) position of the move.
            action_to (Tuple[int, int]): The ending (row, column) position of the move.

        Raises:
            InvalidAction: If jumping over a non-empty space, throne,
                or if improperly jumping over a camp.
        """
        row_from, col_from = action_from
        row_to, col_to = action_to
        board = state.board

        if row_from == row_to:  # Horizontal move
            start, end, fixed = col_from, col_to, row_from
            is_horizontal_move = True
        elif col_from == col_to:  # Vertical move
            start, end, fixed = row_from, row_to, col_from
            is_horizontal_move = False
        else:
            raise InvalidAction("Diagonal moves are not allowed.")

        step = 1 if start < end else -1

        for pos in range(start + step, end, step):
            current_position = (fixed, pos) if is_horizontal_move else (pos, fixed)
            pawn = board.get_piece(current_position)
            is_camp_pos = cls._is_camp(*current_position)

            if pawn != Piece.EMPTY:
                if pawn == Piece.THRONE:
                    raise InvalidAction("Cannot jump over the throne.")
                raise InvalidAction("Cannot jump over a pawn.")
            if is_camp_pos and not cls._is_camp(*action_from):
                raise InvalidAction("Cannot jump over a camp.")

    @classmethod
    def is_valid_move(cls, state: State, move: Action) -> bool:
        """
        Validates if a move follows Tablut's game rules.

        Args:
            state (State): The current game state.
            move (Action): The move to validate.

        Returns:
            bool: True if the move is valid, otherwise raises an exception.

        Raises:
            InvalidAction: If any of the move rules are violated.
        """
        action_from = strp_square(move.from_)
        action_to = strp_square(move.to_)
        row_from, col_from = action_from
        row_to, col_to = action_to
        board_height, board_width = state.board.height, state.board.width
        turn = move.turn
        assert state.turn == move.turn

        if action_to == action_from:
            raise InvalidAction("No movement.")
        if state.board.get_piece(action_from) == Piece.THRONE:
            raise InvalidAction("Cannot move the throne.")
        if state.board.get_piece(action_from) == Piece.EMPTY:
            raise InvalidAction("Nothing to move.")
        is_valid_white_pieces = state.board.get_piece(action_from) not in [Piece.DEFENDER, Piece.KING]
        if turn == Turn.WHITE_TURN and is_valid_white_pieces:
            raise InvalidAction(f"Player {turn} attempted to move opponent's piece in {action_from}.")
        if turn == Turn.BLACK_TURN and state.board.get_piece(action_from) != Piece.ATTACKER:
            raise InvalidAction(f"Player {turn} attempted to move opponent's piece in {action_from}.")

        if col_to >= board_width or row_to >= board_height:
            raise InvalidAction(f"Move {move.to_} is outside board bounds.")
        if action_to == (board_height // 2, board_width // 2):
            raise InvalidAction("Cannot move onto the throne.")
        if cls._is_camp(*action_to) and not cls._is_camp(*action_from):
            raise InvalidAction(f"Cannot enter a camp from outside (from {action_from} to {action_to}).")
        if cls._is_camp(*action_to) and cls._is_camp(*action_from):
            if (row_from == row_to and abs(col_from - col_to) > 5) or (
                    col_from == col_to and abs(row_from - row_to) > 5):
                raise InvalidAction(f"Move from {action_from} to {action_to} exceeds maximum distance within camps.")
        if state.board.get_piece(action_to) != Piece.EMPTY:
            raise InvalidAction(f"Cannot move into a non-empty cell: {action_to}")
        if row_from != row_to and col_from != col_to:
            raise InvalidAction(f"Diagonal moves are not allowed (from {action_from} to {action_to}).")

        cls._check_for_jumps(state, action_from, action_to)
        return True

    @staticmethod
    def __get_all_moves(state: State) -> List[Action]:
        """
        Generates all possible moves for the current turn player based on piece positions.

        Args:
            state (State): The current game state.

        Returns:
            List[Action]: All possible moves in the current state.
        """
        all_actions = []
        turn = state.turn
        board_height, board_width = state.board.height, state.board.width

        if turn == Turn.WHITE_TURN:
            positions = list(
                zip(*np.where((state.board.pieces == Piece.DEFENDER) | (state.board.pieces == Piece.KING))))
        else:
            positions = list(zip(*np.where(state.board.pieces == Piece.ATTACKER)))

        for row, column in positions:
            for index in range(board_height):
                if index != row:
                    all_actions.append(
                        Action(from_=strf_square((row, column)), to_=strf_square((index, column)), turn=turn))
            for index in range(board_width):
                if index != column:
                    all_actions.append(
                        Action(from_=strf_square((row, column)), to_=strf_square((row, index)), turn=turn))

        return all_actions
    
    @staticmethod
    def __gen_all_moves(state: State):
        """
        Generates all possible moves for the current turn player based on piece positions.

        Args:
            state (State): The current game state.

        Returns:
            List[Action]: All possible moves in the current state.
        """
        turn = state.turn
        board_height, board_width = state.board.height, state.board.width

        if turn == Turn.WHITE_TURN:
            positions = list(
                zip(*np.where((state.board.pieces == Piece.DEFENDER) | (state.board.pieces == Piece.KING))))
        else:
            positions = list(zip(*np.where(state.board.pieces == Piece.ATTACKER)))

        for row, column in positions:
            for index in range(board_height):
                if index != row:
                    yield Action(from_=strf_square((row, column)), to_=strf_square((index, column)), turn=turn)
            for index in range(board_width):
                if index != column:
                    yield Action(from_=strf_square((row, column)), to_=strf_square((row, index)), turn=turn)

    @classmethod
    def gen_possible_moves(cls, state: State):
        """
        Filters and returns valid moves from all possible moves for the current player.

        Args:
            state (State): The current game state.

        Returns:
            Generator[Action]: Valid moves for the current game state.
        """
        for move in cls.__gen_all_moves(state):
            try:
                cls.is_valid_move(state, move)
                yield move
            except InvalidAction:
                continue
