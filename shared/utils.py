from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Annotated
from enum import Enum

class Color(Enum):
    """
    Enum representing the colors of the pieces in Tablut.
    """
    WHITE ='W'
    BLACK ='B'
    
class State(BaseModel):
    """
    Model class representing the states of the game in Tablut.
    """
    board: Annotated[Board, "The current state of the game board"]
    turn: Annotated[Color, "The turn player color"]
    

class AbstractPlayer(ABC):
    """
    Abstract base class for players in Tablut.
    """
    
    @property
    @abstractmethod
    def current_state(self):
        pass
    
    @current_state.setter
    @abstractmethod
    def current_state(self, new_state):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def color(self) -> Color:
        pass
    
    @abstractmethod
    def send_move(self):
        pass
    
    @abstractmethod
    def fit(self, state, *args, **kwargs):
        pass