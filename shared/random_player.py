from .utils import AbstractPlayer, Color, State, _Action
from .move_checker import MoveChecker
import random

class RandomPlayer(AbstractPlayer):
    
    def __init__(self, color: Color, initial_state: State = None):
        self.__current_state = initial_state
        self.__name = f'RandomPlayer_{color.value}'
        self.__color = color
    
    def send_move(self) -> None:
        pass
    
    def fit(self, state, *args, **kwargs) -> _Action:
        possible_moves = MoveChecker.get_possible_moves(state)
        return random.choice(possible_moves)