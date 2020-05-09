from keras.models import load_model
from game.board import Board
import numpy as np
import os


class TrainedAgent:
    
    def __init__(self, model_number='latest'):
        filename = 'trained_models/model'
        ext = '.h5'
        if model_number == 'latest':
            n = 0
            while(os.path.exists(filename + str(n) + ext)):
                n += 1
            n -= 1
            model_number = str(n)
        
        self.model = load_model(filename + model_number + ext)
    
    
    def play(self, board: Board):
        possible_boards = board.get_next()
        possible_states = []
        for b in possible_boards:
            possible_states.append(Board.get_game_state(b))
        if len(possible_boards) == 0:
            board.reset_board()
            return

        q_values = []
        for s in possible_states:
            s = np.reshape(s, (1, -1))
            q_values.append(self.model.predict(s)[0])
        m = np.argmax(q_values)

        best_board = possible_boards[m]
        board.do_move(best_board)


