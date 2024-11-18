from pydantic import BaseModel, json
from typing import List, Tuple, Annotated, Optional
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
            f"White Player: {self.white_player}\n"
            f"Black Player: {self.black_player}\n"
            f"Turns:\n{turns_str}\n"
            f"Outcome: {self.outcome}"
        )

class History:
    """
    Class representing the history of matches played.

    Attributes:
        matches (dict[int, Match]): A dictionary mapping match IDs to Match objects.
    """

    def __init__(self):
        """
        Initializes a new History object with an empty dictionary of matches.
        """
        self.matches = {}

    def update_history(self, match_id: int, state: State, action: Action, reward: float):
        """
        Updates the history with a new match, adding the match ID, state, action, and reward.

        Args:
            match_id (int): The unique identifier for the match.
            state (State): The current state of the game.
            action (Action): The action taken by the player.
            reward (float): The reward received for the action.
        """
        if match_id not in self.matches:
            self.matches[match_id] = Match(
                match_id=match_id,
                white_player=RandomPlayer(color=Color.WHITE),
                black_player=RandomPlayer(color=Color.BLACK),
                turns=[],
                outcome=None
            )
        self.matches[match_id].turns.append((state, action, reward))

    def dump(self) -> str:
        return json.dumps(
            {match_id: match.json() for match_id, match in self.matches.items()},
            indent=4
        )
