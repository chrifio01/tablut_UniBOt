"""
This module defines custom exceptions for the Tablut game.

It includes the `InvalidAction` exception, which is used to handle situations 
where a player attempts an illegal move or action. This exception is raised when 
a move violates the game rules, such as moving a piece incorrectly or attempting 
to perform an invalid action.

Classes:
    - InvalidAction: Exception raised for invalid moves or actions in the game.
"""

class InvalidAction(Exception):
    """
    Exception raised when an invalid action is attempted in the game.

    This custom exception is used to signal that a move or action is not allowed 
    according to the game's rules. It is intended to be raised when a player 
    attempts to make an illegal move or when an action violates the game's constraints.
    """
