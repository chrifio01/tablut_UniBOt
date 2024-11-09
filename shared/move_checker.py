from .utils import State, Action
from typing import List

class MoveChecker:
    
    @staticmethod
    def is_valid_move(state: State, move: Action) -> bool:
        pass
    
    @staticmethod
    def __get_all_moves(state: State) -> List[Action]:
        pass
    
    @classmethod
    def get_possible_moves(cls, state: State) -> List[Action]:
        return list(filter(cls.is_valid_move, cls.__get_all_moves(state)))