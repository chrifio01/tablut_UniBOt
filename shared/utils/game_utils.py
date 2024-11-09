from enum import Enum

__all__ = ['Color', 'Piece']

class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.
    """
    WHITE ='W'
    BLACK ='B'
    
class Piece(Enum):
    """
    Enum representing the pieces in Tablut.
    """
    DEFENDER = 'W'
    ATTACKER = 'B'
    KING = 'K'
    THRONE = 'T'
    EMPTY = 'O'