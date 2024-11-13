############################################### Definition of the functions for the evaluation of the Fitness in the heuristic ###########################################################################

import math
import copy
from .game_utils import Board, Piece


def king_distance_from_center(king):
    return ((king[0] - 4)**2 + (king[1] - 4)**2)**0.5


def king_surrounded(board):
    king = board.king_pos()
    c = 0
    blocked_pos = []
    try:
        if board.get_piece()[king[0]+1][king[1]] == Piece.ATTACKER:
            c += 1
            blocked_pos.append((king[0]+1, king[1]))
    except:
        pass
    try:
        if board.get_piece()[king[0]-1][king[1]] == Piece.ATTACKER:
            c += 1
            blocked_pos.append((king[0]-1, king[1]))
    except:
        pass
    try:
        if board.get_piece()[king[0]][king[1]+1] == Piece.ATTACKER:
            c += 1
            blocked_pos.append((king[0], king[1]+1))
    except:
        pass
    try:
        if board.get_piece()[king[0]][king[1]-1] == Piece.ATTACKER:
            c += 1
            blocked_pos.append((king[0], king[1]-1))
    except:
        pass
    return c, blocked_pos


weights = [[0, 20, 20, -6, -6, -6, 20, 20, 0],
           [20, 1, 1, -5, -6, -5, 1,  1, 20],
           [20, 1, 4,  1, -2,  1, 4,  1, 20],
           [-6, -5, 1,  1,  1,  1, 1, -5, -6],
           [-6, -6, -2,  1,  2,  1, -2, -6, -6],
           [-6, -5, 1,  1,  1,  1, 1, -5, -6],
           [20, 1, 4,  1, -2,  1, 4,  1, 20],
           [20, 1, 1, -5, -6, -5, 1,  1, 20],
           [0, 20, 20, -6, -6, -6, 20, 20, 0]]
#

def position_weight(king):
    global weights
    return weights[king[0]][king[1]]


def pawns_around(board, pawn, distance: int):
    """
    Returns the number of pawns around a given pawn within a certain distance (usually the king)
    """
    x, y = pawn
    count = 0
    for i in range(-distance, distance+1):
        for j in range(-distance, distance+1):
            if i == 0 and j == 0:
                continue
            if (x+i, y+j) in board.get_black_coordinates():
                count += 1
    return count