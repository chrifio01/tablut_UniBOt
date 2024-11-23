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

from typing import List, Tuple, Union, Optional, Dict
from pydantic import BaseModel
from shared.utils import Action, State, Turn

class Match(BaseModel):
    """
    Model representing a match in the Tablut game.

    Attributes:
        white_player (str): The player name playing as white.
        black_player (str): The player name playing as black.
        turns (List[Tuple[State, Action, float]]): The list of turns taken during the match, each containing a state, an action, and a reward.
        outcome (Optional[Turn]): The outcome of the match.
    """
    white_player: str
    black_player: str
    turns: List[Tuple[State, Union[Action, None], Union[float, None]]]
    outcome: Optional[Turn]

    class Config:
        """
        Configuration class for the Pydantic model.

        Attributes:
            arbitrary_types_allowed (bool): Allows the model to accept arbitrary types.
        """
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """
        Returns a human-readable string representing the match.

        Returns:
            str: A string with match details.
        """
        turns_str = "\n\t\t\t".join(
            f"Turn {i + 1}:\n___\nState:\n{state}\nAction:\n{action}\nReward: {reward}"
            for i, (state, action, reward) in enumerate(self.turns)
        )
        return (
            "\n_________________________________________\n"
            f"White Player: {self.white_player}\n"
            f"Black Player: {self.black_player}\n"
            f"Turns:\n___________\n{turns_str}\n___________\n"
            f"Outcome: {self.outcome}\n"
            "_________________________________________\n"
        )

class History(BaseModel):
    """
    Class representing the history of matches played.

    Attributes:
        matches (dict[int, Match]): A dictionary mapping match IDs to Match objects.
    """
    matches: Dict[str, Match]

    class Config:
        """
        Configuration class for the Pydantic model.

        Attributes:
            arbitrary_types_allowed (bool): Allows the model to accept arbitrary types.
        """
        arbitrary_types_allowed = True

    def update_history(self, match_id: str, white_player: str, black_player: str, state: State, action: Action, reward: float):
        """
        Updates the history with a new match, adding the match ID, state, action, and reward.

        Args:
            match_id (str): The unique identifier for the match.
            state (State): The current state of the game.
            action (Action): The action taken by the player.
            reward (float): The reward received for the action.
            white_player (str): The player playing as white.
            black_player (str): The player playing as black.
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

    def set_outcome(self, match_id: str, outcome: Turn):
        """
        Sets the outcome of a match.

        Args:
            match_id (str): The unique identifier for the match.
            outcome (Turn): The outcome of the match, which can be Turn.DRAW, Turn.BLACK_WIN, or Turn.WHITE_WIN.

        Raises:
            ValueError: If the outcome is invalid or the match ID is not found.
        """
        if outcome not in (Turn.DRAW, Turn.BLACK_WIN, Turn.WHITE_WIN):
            raise ValueError("Invalid outcome!")
        if match_id not in self.matches:
            raise ValueError("Match not found!")
        self.matches[match_id].outcome = outcome

    def __str__(self) -> str:
        """
        Returns a human-readable string representing the history of matches.

        Returns:
            str: A string with details of all matches.
        """
        string = "Matches:\n"
        for match_id, match in self.matches.items():
            string += f"\t{match_id}:\n{match}"
        return string
