from .blocks import *
from .score import Score
import numpy as np
import random
from copy import deepcopy

class Board:
    _blocks = [[0 for _ in range(10)] for _ in range(20)]
    _block = None

    def generate_block(self):
        block_type = np.random.randint(0, 7)
        if block_type == 0:
            self._block = IBlock(self._blocks)
        elif block_type == 1:
            self._block = OBlock(self._blocks)
        elif block_type == 2:
            self._block = TBlock(self._blocks)
        elif block_type == 3:
            self._block = SBlock(self._blocks)
        elif block_type == 4:
            self._block = ZBlock(self._blocks)
        elif block_type == 5:
            self._block = JBlock(self._blocks)
        else:
            self._block = LBlock(self._blocks)


    def _clear_block(self):
        points = self._block.get_points()
        for p in points:
            row = p[1]
            col = p[0]
            self._blocks[row][col] = 0


    def _save_block(self, b_type):
        points = self._block.get_points()
        for p in points:
            row = p[1]
            col = p[0]
            self._blocks[row][col] = b_type


    def __init__(self):
        self.generate_block()
        self._save_block(self._block.block_type)


    def block_rotate(self):
        self._clear_block()
        self._block.rotate()
        self._save_block(self._block.block_type)


    def drop_block(self):
        self._clear_block()
        while self._block.down():
            pass
        self._save_block(self._block.block_type)
        rows_cleared = Board.clear_rows(self._blocks)
        self.generate_block()
        if not Block.in_bounds(self._block.get_points(), self._blocks):
            return -1
        return rows_cleared

    
    def drop_block_no_gen(self):
        while self._block.down():
            pass
        self._save_block(self._block.block_type)


    def block_down(self):
        self._clear_block()
        rows_cleared = 0
        if not self._block.down():
            self._save_block(self._block.block_type)
            rows_cleared = Board.clear_rows(self._blocks)
            self.generate_block()
            if not Block.in_bounds(self._block.get_points(), self._blocks):
                return -1
        self._save_block(self._block.block_type)
        return rows_cleared

    @staticmethod
    def clear_rows(board):
        cur = 19
        rows_cleared = 0
        while cur >= 0:
            if 0 not in board[cur]:
                rows_cleared += 1
                board = [[0 for _ in range(10)]] + board[:cur] + board[cur+1:] 
            else:
                cur -= 1
        
        return rows_cleared, board


    def block_left(self):
        self._clear_block()
        self._block.left()
        self._save_block(self._block.block_type)


    def block_right(self):
        self._clear_block()
        self._block.right()
        self._save_block(self._block.block_type)


    def get_board(self):
        return self._blocks

    
    def reset_board(self):
        self._blocks = [[0 for _ in range(10)] for _ in range(20)]
        self.generate_block()
        self._save_block(self._block.block_type)


    def get_next(self):
        states = list()
        self._clear_block()
        for _ in range(self._block.rotations):
            for i in range(10):
                self._block.center_at(i, 2)
                if not Block.in_bounds(self._block.get_points(), self._blocks):
                    continue
                self.drop_block_no_gen()

                s = deepcopy(self._blocks)
                states.append(s)
                self._clear_block()
            self._block.rotate_hard()
        
        return states

    
    def do_move(self, board):
        rows_cleared, board = Board.clear_rows(board)
        self._blocks = board
        reward = Score.row_score(rows_cleared)
        self.generate_block()
        game_over = False
        if not Block.in_bounds(self._block.get_points(), self._blocks):
            game_over = True
        return reward, game_over

    
    @staticmethod
    def get_game_state(board):
        heights = list()
        gaps = 0
        for col in range(10):
            current_gap = 0
            h = 20
            row = 19
            while row >= 0:
                if board[row][col] != 0:
                    gaps += current_gap
                    current_gap = 0
                    h = row
                else:
                    current_gap += 1
                row -= 1
            h = 20 - h
            heights.append(h)

        complete_rows = 0
        for row in board:
            if 0 not in row:
                complete_rows += 1

        heights.append(gaps)
        heights.append(complete_rows)
        state = np.array(heights)
        return state

