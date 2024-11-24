"""
This module defines an `AbstractPlayer` base class for implementing player agents in Tablut.

Classes:
    AbstractPlayer: An abstract base class providing a structure for player agents, requiring
    implementations of move generation and model fitting.

Attributes:
    current_state (State): Represents the current game state for the player.
    name (str): Name of the player.
    color (Color): The color of the player (WHITE or BLACK).
"""

from abc import ABC, abstractmethod
from .game_utils import Color, Action
from .env_utils import State


class AbstractPlayer(ABC):
    """
    Abstract base class for players in Tablut, defining required properties and methods for
    interacting with the game environment.

    Attributes:
        current_state (State): Stores the current game state visible to the player.
        name (str): The name or identifier of the player.
        color (Color): The color associated with the player, either WHITE or BLACK.
    
    Methods:
        send_move(): Sends the player's chosen move to the game.
        fit(state, *args, **kwargs) -> Action: Returns the best action determined by the player 
            based on the current state.
    """

    def __init__(self):
        self._current_state = None
        self._name = ""
        self._color = None

    @property
    def current_state(self) -> State:
        """
        Retrieves the current game state for the player.

        Returns:
            State: The current state of the game as observed by the player.
        """
        return self._current_state

    @current_state.setter
    def current_state(self, new_state: State) -> None:
        """
        Sets a new game state for the player.

        Args:
            new_state (State): The new state of the game for the player.
        """
        self._current_state = new_state

    @property
    def name(self) -> str:
        """
        Retrieves the player's name.

        Returns:
            str: The name or identifier of the player.
        """
        return self._name

    @property
    def color(self) -> Color:
        """
        Retrieves the color assigned to the player.

        Returns:
            Color: The color of the player, either WHITE or BLACK.
        """
        return self._color
    
    @color.setter
    def color(self, new_color: Color) -> None:
        self._color = new_color

    @abstractmethod
    def fit(self, state: State, *args, **kwargs) -> Action:
        """
        Abstract method to calculate the optimal move based on the current state.

        Args:
            state (State): The current game state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Action: The action chosen by the player based on the game state.
        """
