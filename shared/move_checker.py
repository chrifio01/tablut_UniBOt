from .utils import State, _Action, strf_square
from typing import List, Tuple
import numpy as np

class MoveChecker:
    
    @staticmethod
    def is_valid_move(state: State, move: _Action) -> bool:
        pass
    
    @staticmethod
    def __get_all_moves(state: State) -> List[_Action]:
        all_actions = []
        turn = state.turn
        positions_of_movable_pieces: List[Tuple[int, int]] = list(zip(*(np.where(state.board.pieces == turn))))
        
        board_height = state.board.height
        board_width = state.board.width
        
        for column, row in positions_of_movable_pieces:
            for index in range(0, board_height + 1):
                if index == row:
                    continue
                vertical_action = _Action(from_=strf_square((column, row)), to_=strf_square((column, index)), turn=turn)
                all_actions.append(vertical_action)
                
            for index in range(0, board_width + 1):
                if index == column:
                    continue
                horizontal_action = _Action(from_=strf_square((column, row)), to_=strf_square((index, row)), turn=turn)
                all_actions.append(horizontal_action)
                
        return all_actions
    
    @classmethod
    def get_possible_moves(cls, state: State) -> List[_Action]:
        return list(filter(cls.is_valid_move, cls.__get_all_moves(state)))