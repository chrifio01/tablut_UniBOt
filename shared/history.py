"""
This module defines the `History` and `Match` classes for managing the history of matches played in the Tablut game.

Classes:
    Match: Represents a match in the Tablut game, including players, turns, and outcome.
    History: Manages the history of matches, allowing updates and serialization.

Attributes:
    Match.match_id (int): The unique identifier for the match.
    Match.white_player (AbstractPlayer): The player playing as white.
    Match.black_player (AbstractPlayer): The player playing as black.
    Match.turns (List[Tuple[State, Action, float]]): The list of turns taken during the match, each containing a state, an action, and a reward.
    Match.outcome (Optional[Turn]): The outcome of the match.
    History.matches (dict[int, Match]): A dictionary mapping match IDs to Match objects.

Methods:
    Match.__str__: Returns a human-readable string representing the match.
    History.__init__: Initializes a new History object with an empty dictionary of matches.
    History.update_history: Updates the history with a new match, adding the match ID, state, action, and reward.
    History.dump: Serializes the history of matches to a JSON string.
"""

from typing import List, Tuple, Annotated, Optional, Dict
from pydantic import BaseModel
from shared.random_player import RandomPlayer
from shared.utils import AbstractPlayer, Action, State, Turn, Color

class Match(BaseModel):
    """
    Model representing a match in the Tablut game.

    Attributes:
        match_id (int): The unique identifier for the match.
        white_player (AbstractPlayer): The player playing as white.
        black_player (AbstractPlayer): The player playing as black.
        turns (List[Tuple[State, Action, float]]): The list of turns taken during the match, each containing a state, an action, and a reward.
        outcome (Optional[Turn]): The outcome of the match.
    """
    match_id: Annotated[int, "The unique identifier for the match"]
    white_player: Annotated[AbstractPlayer, "The player playing as white"]
    black_player: Annotated[AbstractPlayer, "The player playing as black"]
    turns: Annotated[List[Tuple[State, Action, float]], "The list of turns taken during the match, each containing a state, an action, and a reward"]
    outcome: Annotated[Optional[Turn], "The outcome of the match"]

    def __str__(self) -> str:
        """
        Returns a human-readable string representing the match.

        Returns:
            str: A string with match details.
        """
        turns_str = "\n".join(
            f"Turn {i + 1}:\nState:\n{state}\nAction: {action}\nReward: {reward}"
            for i, (state, action, reward) in enumerate(self.turns)
        )
        return (
            f"Match ID: {self.match_id}\n"
            f"White Player: {self.white_player.name}\n"
            f"Black Player: {self.black_player.name}\n"
            f"Turns:\n{turns_str}\n"
            f"Outcome: {self.outcome}"
        )

class History(BaseModel):
    """
    Class representing the history of matches played.

    Attributes:
        matches (dict[int, Match]): A dictionary mapping match IDs to Match objects.
    """
    matches: Annotated[Dict[int, Match], "A dictionary mapping match IDs to Match objects"]

    def update_history(self, match_id: int, white_player: AbstractPlayer, black_player: AbstractPlayer, state: State, action: Action, reward: float):
        """
        Updates the history with a new match, adding the match ID, state, action, and reward.

        Args:
            match_id (int): The unique identifier for the match.
            state (State): The current state of the game.
            action (Action): The action taken by the player.
            reward (float): The reward received for the action.
            white_player (AbstractPlayer): The player playing as white.
            black_player (AbstractPlayer): The player playing as black.
        """
        if match_id not in self.matches:
            self.matches[match_id] = Match(
                match_id=match_id,
                white_player=white_player,
                black_player=black_player,
                turns=[],
                outcome=None
            )
        self.matches[match_id].turns.append((state, action, reward))
