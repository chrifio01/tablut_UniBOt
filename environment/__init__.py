"""
This module initializes the environment package for the Tablut game.

It imports the `Environment` class from the `tablut` module, which is used to
simulate the game environment, handle game logic, and manage the state of the game.

Classes:
    Environment: Represents the game environment for Tablut.
"""
from .tablut import Environment
from .utils import state_to_tensor, ActionDecoder
