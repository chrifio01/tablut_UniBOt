from .utils import State, _Action, strf_square, strp_square, Piece, Color
from typing import List, Tuple
from .consts import CAMPS
from .exceptions import InvalidAction
import numpy as np

class MoveChecker:
    
    @staticmethod
    def __is_camp(row: int, col: int) -> bool:
        """Returns True if the position is a camp."""
        return (row, col) in CAMPS

    @classmethod
    def __check_for_jumps(cls, state: State, action_from: Tuple[int, int], action_to: Tuple[int, int]) -> None:
        """
        Checks for illegal jumps over non-empty spaces, the throne, or citadels during a move.

        Args:
            state (State): The current game state.
            action_from (Tuple[int, int]): The starting (row, column) position of the move.
            action_to (Tuple[int, int]): The ending (row, column) position of the move.

        Raises:
            InvalidAction: If jumping over a non-empty space, throne, or if improperly jumping over a citadel.
        """
        row_from, col_from = action_from
        row_to, col_to = action_to
        board = state.board

        # Determine the direction of movement
        if row_from == row_to:  # Horizontal move
            start, end = col_from, col_to
            fixed = row_from
            is_horizontal_move = True
        elif col_from == col_to:  # Vertical move
            start, end = row_from, row_to
            fixed = col_from
            is_horizontal_move = False
        else:
            raise InvalidAction("Diagonal moves are not allowed.")

        # Define step for traversal
        step = 1 if start < end else -1

        # Traverse between start and end in the specified direction
        for pos in range(start + step, end, step):
            current_position = (fixed, pos) if is_horizontal_move else (pos, fixed)
            pawn = board.get_piece(current_position)
            is_camp_pos = cls.__is_camp(*current_position)

            # Check if jumping over non-empty spaces or throne
            if pawn != Piece.EMPTY:
                if pawn == Piece.THRONE:
                    raise InvalidAction("Cannot jump over the throne.")
                raise InvalidAction("Cannot jump over a pawn.")

            # Check for improper citadel jumps
            if is_camp_pos and not cls.__is_camp(*action_from):
                raise InvalidAction("Cannot jump over a camp.")

    @classmethod
    def is_valid_move(cls, state: State, move: _Action) -> bool:
        """
        Validates if a given move complies with game rules.

        Args:
            state (State): The current game state.
            move (_Action): The move to validate.

        Returns:
            bool: True if the move is valid, False otherwise.

        Raises:
            InvalidAction: If any of the move rules are violated.
        """
        # Extract row and column information for the action
        action_from = strp_square(move.from_)
        action_to = strp_square(move.to_)
        board_height = state.board.height
        board_width = state.board.width
        row_from, col_from = action_from
        row_to, col_to = action_to
        turn = state.turn
        
        # Check if move does not change position
        if action_to == action_from:
            raise InvalidAction("No movement.")
        
        # Check if is trying to move the throne
        if state.board.get_piece(action_from) == Piece.THRONE:
            raise InvalidAction("Cannot move the throne.")

        # Ensure player moves only their own pieces
        if turn == Color.WHITE:
            if state.board.get_piece(action_from) not in [Piece.DEFENDER, Piece.KING]:
                raise InvalidAction(f"Player {turn} attempted to move opponent's piece in {action_from}.")
        elif turn == Color.BLACK:
            if state.board.get_piece(action_from) != Piece.ATTACKER:
                raise InvalidAction(f"Player {turn} attempted to move opponent's piece in {action_from}.")

        # Check if move is within board bounds
        if col_to >= board_width or row_to >= board_height:
            raise InvalidAction(f"Move {move.to_} is outside board bounds.")

        # Check for throne (center position in 9x9 Ashton Tablut is (4,4))
        if action_to == (board_height // 2, board_width // 2):
            raise InvalidAction("Cannot move onto the throne.")

        # Camp entry check: disallow moves from outside into a camp
        if cls.__is_camp(*action_to) and not cls.__is_camp(*action_from):
            raise InvalidAction(f"Cannot enter a camp from outside (from {action_from} to {action_to}).")

        # Long-distance camp move check
        if cls.__is_camp(*action_to) and cls.__is_camp(*action_from):
            if (row_from == row_to and abs(col_from - col_to) > 5) or (col_from == col_to and abs(row_from - row_to) > 5):
                raise InvalidAction(f"Move from {action_from} to {action_to} exceeds maximum distance within camps.")

        # No diagonal moves allowed
        if row_from != row_to and col_from != col_to:
            raise InvalidAction(f"Diagonal moves are not allowed (from {action_from} to {action_to}).")

        # Check for invalid jumps
        cls.__check_for_jumps(state, action_from, action_to)

        # If all checks are passed, the move is valid
        return True
    
    @staticmethod
    def __get_all_moves(state: State) -> List[_Action]:
        all_actions = []
        turn = state.turn
        positions_of_movable_pieces: List[Tuple[int, int]] = None
        
        if turn == Color.WHITE:
            # Find both DEFENDER and KING pieces for the white turn
            positions_of_movable_pieces = list(
                zip(*np.where((state.board.pieces == Piece.DEFENDER) | (state.board.pieces == Piece.KING)))
            )
        else:
            # Only find ATTACKER pieces for the black turn
            positions_of_movable_pieces = list(
                zip(*np.where(state.board.pieces == Piece.ATTACKER))
            )
    
        board_height = state.board.height
        board_width = state.board.width
        
        for row, column in positions_of_movable_pieces:
            for index in range(0, board_height):
                if index == row:
                    continue
                vertical_action = _Action(from_=strf_square((row, column)), to_=strf_square((index, column)), turn=turn)
                all_actions.append(vertical_action)
                
            for index in range(0, board_width):
                if index == column:
                    continue
                horizontal_action = _Action(from_=strf_square((row, column)), to_=strf_square((row, index)), turn=turn)
                all_actions.append(horizontal_action)
                
        return all_actions
    
    @classmethod
    def get_possible_moves(cls, state: State) -> List[_Action]:
        return list(filter(cls.is_valid_move, cls.__get_all_moves(state)))