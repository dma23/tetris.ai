from keras.models import Sequential
from keras.layers import Dense
from .memory import ReplayMemory
from game.board import Board
import os
import numpy as np
from keras.models import load_model
from keras import Sequential

class Agent:

    def __init__(self, load_previous=False):
        # things that need to be initialized here:
        # replay memory
        # hyperparameters
        #   min epsilon
        #   gamma or discount factor
        #   learning rate (this is sort of replaced by discount factor in reinforcement learning)
        #   epislon decay factor
        # model
        """
        self.epsilon = 1
        self.memory = ReplayMemory()
        self.model = self.build_model()
        self.gamma = 0.95
        self.episodes = 2048
        self.decay = 0
        # self.visualizer = Visualizer()
        """

        # model12 is the last iteration before re-learning begins
        # 19 is ok as well
        # 31 is ok 
        # 33 
        self.model = load_model('trained_models/model36.h5')
        self.epsilon = 0.035
        self.decay = 0
        self.episodes = 4056
        self.gamma = 0.95
        self.memory = ReplayMemory()

    def build_model(self):
        # before the model is initialized, we need to decide on how we will be representing a state:
        #   the entire board
        #   the heights of each column
        #   human heuristics from playing tetris (number of holes, bumps, etc)
        # we can try using different combinations of these in which case build_model and init
        # would take extra parameters to indicate the model we want to use for the current instance

        # probably initialize model with standard 2 hidden layer architecture here
        # 2 hidden layers (32 to 64 neurons in each layer)
        # 1 output neuron (estimate the q - value of the board)
        # no point in using dropout layers since chance of overfit is almost 0 and we wouldnt
        # know when it is overfitting anyways
        model = Sequential()
        model.add(Dense(512, input_shape=(12,), activation = 'relu'))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dense(1, activation = 'linear'))
        model.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')
        # for activation functions use reLU for hidden layers
        # we don't want to use sigmoid since there is no real point of restricting between 0 and 1
        # choice of activation function for output layer:
        # linear - if we want a full range of q - values (negative/positive and unbounded)
        # reLU - if we want only positive unbounded values (positive)
        # sigmoid - if we want only values between 0 and 1
        # probably use either reLU or linear
        return model

    def predict(self, board: Board):
        possible_boards = board.get_next()
        possible_states = []

        for b in possible_boards:
            possible_states.append(Board.get_game_state(b))
        # this will return a prediction based on the current state of the game
        # the prediction will be one of:
        #   random action if random val < epsilon
        #   best action if random val >= epsilon
        if len(possible_boards) == 0:
            return -1, np.zeros((11,))
       
        # make a random move
        if np.random.random() < self.epsilon:
            m = np.random.randint(0, len(possible_boards))
            return possible_boards[m], possible_states[m]

        # make the best possible move
        else:
            q_values = []
            for s in possible_states:
                s = np.reshape(s, (1, -1))
                q_values.append(self.model.predict(s)[0])
            m = np.argmax(q_values)
        return possible_boards[m], possible_states[m]



    def train(self):
        # this is where the model will train itself
        # the model should perform random actions until there is sufficient data in replay memory
        # once there is, the model will also train off replay memory after every action
        mem_size = len(self.memory.memories)
        if mem_size < self.memory.mem_size:
            return

        mems = self.memory.sample()
        punishment = -1

        next_states = [mem[0] for mem in mems]
        q_values = list()
        for state in next_states:
            state = np.reshape(state, (1, -1))
            q_values.append(self.model.predict(state)[0])

        inputs = list()
        targets = list()

        for i in range(len(mems)):
            current_state = mems[i][0]
            reward = mems[i][2]
            game_over = mems[i][3]

            q_prime = reward + self.gamma * q_values[i] if not game_over else punishment

            inputs.append(current_state)
            targets.append(q_prime)

        inputs = np.array(inputs)
        targets = np.array(targets)
        self.model.fit(x=inputs, y=targets, epochs=4, batch_size=64, verbose=0)


    def run(self, board: Board):
        sub100 = 0
        max_score = 0
        total = 0

        for e in range(self.episodes):
            print('Iteration: ', e)
            board.reset_board()
            
            game_over = False
            total_score = 0
            #print('Memories', len(self.memory.memories))
            #print('Epsilon', self.epsilon)
            while not game_over:
                current_state = Board.get_game_state(board._blocks)
                new_board, new_state = self.predict(board)

                reward = 0
                if new_board == -1:
                    game_over = True
                else:
                    reward, game_over = board.do_move(new_board)

                total_score += reward

                self.memory.remember((current_state, new_state, reward, game_over))

            print('Score', total_score)
            self.train()
            self.epsilon -= self.decay

            total += total_score
            max_score = total_score if total_score > max_score else max_score
            if total_score < 100:
                sub100 += 1
        
        # generate summary
        print('Max score', max_score)
        print('Avg score', total/self.episodes)
        print('Sub 100', sub100)
        self.save()


    def save(self):
        filename = 'trained_models/model'
        ext = '.h5'
        n = 0
        while(os.path.exists(filename + str(n) + ext)):
            n += 1
        self.model.save(filename + str(n) + ext)
        print('Saved to', filename + str(n) + ext)
        pass

