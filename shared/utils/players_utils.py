from abc import ABC, abstractmethod
from .game_utils import Color

class AbstractPlayer(ABC):
    """
    Abstract base class for players in Tablut.
    """
    
    @property
    def current_state(self):
        return self.__current_state
    
    @current_state.setter
    def current_state(self, new_state):
        self.__current_state = new_state
    
    @property
    def name(self):
        return self.__name
    
    @property
    def color(self) -> Color:
        return self.__color
    
    @abstractmethod
    def send_move(self):
        pass
    
    @abstractmethod
    def fit(self, state, *args, **kwargs):
        pass