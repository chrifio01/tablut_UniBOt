from abc import ABC, abstractmethod
from .game_utils import Color, _Action
from .env_utils import State

class AbstractPlayer(ABC):
    """
    Abstract base class for players in Tablut.
    """
    
    @property
    def current_state(self) -> State:
        return self.__current_state
    
    @current_state.setter
    def current_state(self, new_state) -> None:
        self.__current_state = new_state
    
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def color(self) -> Color:
        return self.__color
    
    @abstractmethod
    def send_move(self) -> None:
        pass
    
    @abstractmethod
    def fit(self, state, *args, **kwargs) -> _Action:
        pass