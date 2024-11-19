
from shared.utils.game_utils import Board, Action
from shared.utils.env_utils import State, pawns_around
from shared.history import History
from pydantic import BaseModel
from typing import List, Tuple, Annotated, Optional, Dict
from shared.heuristic import heuristic



class Environment(BaseModel):

    board: Annotated[Board, "A class Board object"]
    currentState: Annotated[State, "The current config of the board piecese, and player tourn"]
    historyUpdater: Annotated[History, "Past states and moves"]

    def is_it_a_tie(self)->bool:
        pass

    def did_black_WIN(self)->bool:
        if pawns_around(self.board, self.board.king_pos, 1) == 4:
            return True
        return False
        
    def did_white_WIN(self)->bool:
        win_tiles = [(0,1),(0,2),(0,6),(0,7),(1,0),(2,0),(6,0),(7,0),(8,1),(8,2),(8,6),(8,7),(1,8),(2,8),(6,8),(7,8)]
        if self.board.king_pos() in win_tiles:
            return True
        return False
    
    def get_winnner(self):
        if self.did_black_WIN():
            return "BW"
        if self.did_white_WIN():
            return "WW"
        return None
    
    def calculate_rewards(self, move: Action):
        return heuristic(self.currentState, move)
    
    def update_state(self, move: Action):
        self.board.update_pieces(move)
        pass
        

