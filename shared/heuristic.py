import math
from shared.utils.env_utils import king_distance_from_center, king_surrounded, position_weight, pawns_around, strp_state, State
from shared.utils.game_utils import Action, Board, Color, strp_board
from .move_checker import MoveChecker
from .consts import *
import copy


def _white_heuristic(board: Board):
    """
    Returns the float value of the current state of the board for white
    """
    
    

    king_pos = board.king_pos() 

        

    #alpha0, beta0, gamma0, theta0, epsilon0, omega0 = [
    #    12, 22, 9, 1, 2, 20]

    fitness = 0 

    # Blackpieces
    num_blacks = board.num_black()
    fitness -= ALPHA_W * num_blacks

    # whitepieces
    num_whites = board.num_white()
    fitness += BETA_W * num_whites

    # king distance
    fitness += king_distance_from_center(board, king_pos) * GAMMA_W

    # free ways
    free_paths = [board.is_there_a_clear_view(black_pawn, king_pos)
                for black_pawn in board.get_black_coordinates()]
    # theta0 times the n° free ways to king
    fitness -= OMEGA_W * sum(free_paths)

    # king surrounded
    king_vals, _ = king_surrounded(board)
    fitness -= king_vals * THETA_W

    fitness += position_weight(king_pos) * EPSILON_W # Return maximum values when king is in escape tiles !!WIN CONFIG!! 

    norm_fitness = (
        fitness / (16 * BETA_W + math.sqrt(32) * GAMMA_W + 20*EPSILON_W))

        # print("WHITE FITNESS: ", norm_fitness)

    return fitness
    

def _black_heuristic(board: Board):
    """
    Black heuristics should be based on:
    - Number of black pawns
    - Number of white pawns
    - Number of black pawns next to the king
    - Free path to the king
    - A coefficient of encirclement of the king
    """
        
   

    fitness = 0

        

    king_pos = board.king_pos()

    # Number of black pawns
    num_blacks = board.num_black()
    fitness += ALPHA_B * num_blacks

    # Number of white pawns
    num_whites = board.num_white()
    fitness -= BETA_B * num_whites

    # Number of black pawns next to the king
    fitness += GAMMA_B * pawns_around(board, king_pos, distance=1)  # Maximum value for king surrounded on all 4 sides !!WIN CONFIG!!

    # Free path to the king
    free_paths = [board.is_there_a_clear_view(black_pawn, king_pos)
                for black_pawn in board.get_black_coordinates()]
    # theta0 times the n° free ways to king
    fitness += THETA_B * sum(free_paths)

    # norm_fitness = (fitness / (alpha0 * len(board.blacks) + gamma0 *
    #                           pawns_around(board, king_pos, distance=2) + theta0 * sum(free_paths)))

    # print("BLACK FITNESS: ", norm_fitness)

    return fitness

   

def heuristic(state: State, move: Action):
    """
    Returns the float value of the possible state of the board for the player that has to play, according to the move passed as argument 

    Arg:
    state: a string that represent the current state of the board, with also the turn of the player that has to make a move
    move: a class Action that represent the move on which the heuristic is calculated

    Return:
    float value of the move
    """
    if MoveChecker.is_valid_move(state, move):

        board = Board(state.board.pieces)

        board.update_pieces(move)


        if move.turn == Color.WHITE:
            return _white_heuristic(board) - _black_heuristic(board)
        if move.turn == Color.BLACK:
            return _black_heuristic(board) - _white_heuristic(board)
    else:
        return -9999

