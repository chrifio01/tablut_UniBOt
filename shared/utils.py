from abc import ABC, abstractmethod

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
    def color(self):
        pass
    
    @abstractmethod
    def send_move(self):
        pass
    
    @abstractmethod
    def fit(self, state, *args, **kwargs):
        pass