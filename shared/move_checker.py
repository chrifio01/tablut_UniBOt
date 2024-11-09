from .utils import State, _Action
from typing import List

class MoveChecker:
    
    @staticmethod
    def is_valid_move(state: State, move: _Action) -> bool:
        pass
    
    @staticmethod
    def __get_all_moves(state: State) -> List[_Action]:
        pass
    
    @classmethod
    def get_possible_moves(cls, state: State) -> List[_Action]:
        return list(filter(cls.is_valid_move, cls.__get_all_moves(state)))