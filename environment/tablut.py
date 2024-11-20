
from shared.utils.game_utils import Board, Action, Turn
from shared.utils.env_utils import State, black_win_con
from shared.consts import WIN_TILES
from shared.history import History
from shared.move_checker import MoveChecker
from pydantic import BaseModel
from typing import Annotated
from shared.heuristic import heuristic



class Environment(BaseModel):

    board: Annotated[Board, "A class Board object"]
    currentState: Annotated[State, "The current config of the board piecese, and player tourn"]
    historyUpdater: Annotated[History, "Past states and moves"]

    def is_it_a_tie(self, match_id: int)->bool:
        if self.currentState in self.historyUpdater.matches[match_id].turns:
            return True
        return False

    def did_black_WIN(self, move: Action)->bool:
        if black_win_con(self.board, self.board.king_pos()) == 4:
            return True
        if self.currentState.turn == Turn.WHITE_TURN:      
            if not MoveChecker.is_valid_move(self.currentState, move):
                return True
        return False
        
    def did_white_WIN(self, move: Action)->bool:
        if self.board.king_pos() in WIN_TILES:
            return True
        if self.currentState.turn == Turn.BLACK_TURN:      
            if not MoveChecker.is_valid_move(self.currentState, move):
                return True
        return False
    
    def get_winnner(self):
        if self.did_black_WIN():
            return Turn.BLACK_WIN
        if self.did_white_WIN():
            return Turn.WHITE_WIN
        return None
    
    def calculate_rewards(self, match_id: int):
        return heuristic(self.currentState, self.historyUpdater.matches[match_id].turns[1])
    
    def update_state(self, move: Action):
        self.board.update_pieces(move)
        pass

   

        

