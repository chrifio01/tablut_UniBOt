"""
Module for implementing a random player in the game of Tablut.

The `RandomPlayer` class represents a player that randomly selects a valid move 
from the list of available moves. This player extends the abstract `AbstractPlayer` 
class and overrides the `fit` method to make random decisions for its moves.
"""

from .utils import AbstractPlayer, Color, State, Action
from .move_checker import MoveChecker
import random

class RandomPlayer(AbstractPlayer):
    """
    A player that randomly selects a valid move from the available options.

    Inherits from AbstractPlayer and overrides the `fit` method to select a move at random.
    """

    def __init__(self, color: Color, initial_state: State = None):
        """
        Initializes a RandomPlayer instance.

        Args:
            color (Color): The color of the player (either WHITE or BLACK).
            initial_state (State, optional): The initial game state. Defaults to None.
        """
        self.__current_state = initial_state
        self.__name = f'RandomPlayer_{color.value}'
        self.__color = color
    
    def send_move(self) -> None:
        """
        Placeholder method for sending the move. In this case, the player doesn't need to send anything manually.
        """
        pass
    
    def fit(self, state: State, *args, **kwargs) -> Action:
        """
        Chooses a random valid move from the available possible moves.

        Args:
            state (State): The current game state.
            *args, **kwargs: Additional arguments that might be used in the future (currently unused).

        Returns:
            Action: A random valid action that the player can take based on the current game state.
        """
        possible_moves = MoveChecker.get_possible_moves(state)
        return random.choice(possible_moves)