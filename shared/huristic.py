import math
from shared.utils.env_utils import king_distance_from_center, king_surrounded, position_weight, pawns_around, strp_state
from shared.utils.game_utils import Action, Board, Color, strp_board
from .move_checker import MoveChecker
from .consts import *
import copy


def white_heuristic(state: str, move: Action):
    """
    Returns the float value of the current state of the board for white
    """
    if MoveChecker.is_valid_move(strp_state(state), move):

        splitted_state_str = state.split('-')
        board_st = splitted_state_str[0].strip()
    
        
        board = Board(strp_board(board_st))
    
        board.update_pieces(move)
    

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

        fitness += position_weight(king_pos) * EPSILON_W

        norm_fitness = (
            fitness / (16 * BETA_W + math.sqrt(32) * GAMMA_W + 20*EPSILON_W))

        # print("WHITE FITNESS: ", norm_fitness)

        return fitness
    else:
        return -9999


def black_heuristic(state: str, move: Action):
    """
    Black heuristics should be based on:
    - Number of black pawns
    - Number of white pawns
    - Number of black pawns next to the king
    - Free path to the king
    - A coefficient of encirclement of the king
    """

    if MoveChecker.is_valid_move(strp_state(state), move):

        splitted_state_str = state.split('-')
        board_st = splitted_state_str[0].strip()
        
        board = Board(strp_board(board_st))

        board.update_pieces(move)


        fitness = 0

        

        king_pos = board.king_pos()

        # Number of black pawns
        num_blacks = board.num_black()
        fitness += ALPHA_B * num_blacks

        # Number of white pawns
        num_whites = board.num_white()
        fitness -= BETA_B * num_whites

        # Number of black pawns next to the king
        fitness += GAMMA_B * pawns_around(board, king_pos, distance=1)

        # Free path to the king
        free_paths = [board.is_there_a_clear_view(black_pawn, king_pos)
                    for black_pawn in board.get_black_coordinates()]
        # theta0 times the n° free ways to king
        fitness += THETA_B * sum(free_paths)

    # norm_fitness = (fitness / (alpha0 * len(board.blacks) + gamma0 *
    #                           pawns_around(board, king_pos, distance=2) + theta0 * sum(free_paths)))

    # print("BLACK FITNESS: ", norm_fitness)

        return fitness

    else:
        return -9999


def heuristic(state: str, move: Action):
    """
    Returns the float value of the possible state of the board for the player that has to play, according to the move passed as argument 

    Arg:
    state: a string that represent the current state of the board, with also the turn of the player that has to make a move
    move: a class Action that represent the move on which the heuristic is calculated

    Return:
    float value of the move
    """
    if move.turn == Color.WHITE:
        return white_heuristic(state, move) - black_heuristic(state, move)
    if move.turn == Color.BLACK:
        return black_heuristic(state, move) - white_heuristic(state, move)

