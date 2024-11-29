"""
This module implements heuristic evaluation functions for a board game involving two sides, white and black. 
It defines heuristic functions used to evaluate board states for both players (white and black) based on various 
factors such as the number of pawns, the king's position, and strategic elements like free paths and encirclement.
The main heuristic function integrates move validation and calculates a heuristic score to assist in decision-making.
"""

from shared.utils.env_utils import king_distance_from_center, king_surrounded, position_weight, pawns_around, State
from shared.utils.game_utils import Action, Board, Turn, Piece
from .move_checker import MoveChecker
from .consts import ALPHA_W, BETA_W, GAMMA_W, THETA_W, EPSILON_W, OMEGA_W, ALPHA_B, BETA_B, GAMMA_B, THETA_B, INVALID_ACTION_PUNISHMENT
from .exceptions import InvalidAction



def _white_heuristic(board: Board) -> float:
    """
    Heuristic function for evaluating the state for White.

    Args:
        board (Board): The current game board.

    Returns:
        float: Heuristic score favoring White.
    """
    king_pos = board.king_pos()
    score = 0

    # Reward king's proximity to escape routes
    score += position_weight(king_pos)

    # Penalize threats around the king
    threats = board.num_threats_to_piece(king_pos, Piece.KING)
    score -= threats

    # Reward clear escape routes
    escape_routes = board.king_free_escape_routes(king_pos)
    score += escape_routes

    # Adjust for the number of active white pieces
    score += board.num_white()

    return float(score)


def _black_heuristic(board: Board) -> float:
    """
    Heuristic function for evaluating the state for Black.

    Args:
        board (Board): The current game board.

    Returns:
        float: Heuristic score favoring Black.
    """
    king_pos = board.king_pos()
    score = 0

    # Reward surrounding the king
    threats = board.num_threats_to_piece(king_pos, Piece.KING)
    score += threats

    # Penalize king's clear escape routes
    escape_routes = board.king_free_escape_routes(king_pos)
    score -= escape_routes

    # Adjust for the number of active black pieces
    score += board.num_black() // 2

    return float(score)



def heuristic(state: State, move: Action):
    """
    Returns the float value of the possible state of the board for the player that has to play, according to the move passed as argument 

    Arg:
    state: a string that represent the current state of the board, with also the turn of the player that has to make a move
    move: a class Action that represent the move on which the heuristic is calculated

    Return:
    float value of the move
    """
    try:
        MoveChecker.is_valid_move(state, move)

        board = Board(state.board.pieces)

        board.update_pieces(move)


        if move.turn == Turn.WHITE_TURN:
            return _white_heuristic(board)
        if move.turn == Turn.BLACK_TURN:
            return _black_heuristic(board)
        return None
    except InvalidAction:
        # If the move is invalid, return a very low value to avoid the move from being chosen
        return INVALID_ACTION_PUNISHMENT
