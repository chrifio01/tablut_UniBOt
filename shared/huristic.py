import math
from .utils.state_utils import * 
from .utils import Action
from move_checker import MoveChecker
import copy


def white_fitness(board: Board, action: Action, MC: MoveChecker):
    """
    Returns the float value of the current state of the board for white
    """
    if MC.is_valid_move(board.pieces(), action):
        
        board.update_pieces(action)

        king_pos = board.king_pos() 

        alpha0, beta0, gamma0, theta0, epsilon0, omega0 = [0.21639120828483156, 0.723587137336777, 9, 1.06923818569000507, 2.115749207248323, 10]

        #alpha0, beta0, gamma0, theta0, epsilon0, omega0 = [
        #    12, 22, 9, 1, 2, 20]

        fitness = 0

        # Blackpieces
        num_blacks = board.num_black()
        fitness -= alpha0 * num_blacks

        # whitepieces
        num_whites = board.num_white()
        fitness += beta0 * num_whites

        # king distance
        fitness += king_distance_from_center(king_pos) * gamma0

        # free ways
        free_paths = [board._is_there_a_clear_view(black_pawn, king_pos)
                    for black_pawn in board.get_black_coordinates()]
        # theta0 times the n° free ways to king
        fitness -= omega0 * sum(free_paths)

        # king surrounded
        king_vals, _ = king_surrounded(board)
        fitness -= king_vals * theta0

        fitness += position_weight(king_pos) * epsilon0

        norm_fitness = (
            fitness / (16 * beta0 + math.sqrt(32) * gamma0 + 20*epsilon0))

        # print("WHITE FITNESS: ", norm_fitness)

        return fitness
    else:
        return -9999


def black_fitness(board: Board, action: Action, MC: MoveChecker):
    """
    Black heuristics should be based on:
    - Number of black pawns
    - Number of white pawns
    - Number of black pawns next to the king
    - Free path to the king
    - A coefficient of encirclement of the king
    """
    
    if MC.is_valid_move(board.pieces(), action):

        board.update_pieces(action)
        
        fitness = 0

        alpha0, beta0, gamma0, theta0, epsilon0 = [0.958245251997756, 0.25688393654958275, 0.812052344592159, 0.9193347856045799, 1.7870310915100207]

        king_pos = board.king_pos()

        # Number of black pawns
        num_blacks = board.num_black()
        fitness += alpha0 * num_blacks

        # Number of white pawns
        num_whites = board.num_white()
        fitness -= beta0 * num_whites

        # Number of black pawns next to the king
        fitness += gamma0 * pawns_around(board, king_pos, distance=1)

        # Free path to the king
        free_paths = [board._is_there_a_clear_view(black_pawn, king_pos)
                    for black_pawn in board.get_black_coordinates()]
        # theta0 times the n° free ways to king
        fitness += theta0 * sum(free_paths)

        # norm_fitness = (fitness / (alpha0 * len(board.blacks) + gamma0 *
        #                           pawns_around(board, king_pos, distance=2) + theta0 * sum(free_paths)))

        # print("BLACK FITNESS: ", norm_fitness)

        return fitness
    else:
        return -9999


