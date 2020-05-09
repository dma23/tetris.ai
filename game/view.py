import pyglet
from pyglet.window import key
from .board import Board
from .score import Score
from pyglet import clock
from model.trained_agent import TrainedAgent


class View(pyglet.window.Window):
    # load image assets
    L = pyglet.image.load('images/L.png')
    Z = pyglet.image.load('images/R.png')
    T = pyglet.image.load('images/S.png')
    O = pyglet.image.load('images/T_2.png')
    I = pyglet.image.load('images/1.png')

    # sets some colours
    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 128)
    font = pyglet.font.load('freesansbold.ttf', 12)

    scorer = Score()
    _score = scorer.get_score()
    _level = scorer.get_level()


    _score_label = pyglet.text.Label(
        'Score: ' + str(0), 
        font_name='freesansbold.ttf', 
        font_size=12,
        x=250, y=50,
        anchor_x='center', anchor_y='center')
    
    _level_label = pyglet.text.Label(
        'Level: ' + str(1), 
        font_name='freesansbold.ttf', 
        font_size=12,
        x=250, y=30,
        anchor_x='center', anchor_y='center')

    _board = Board()
    
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trained_agent = TrainedAgent()


    def agent_play(self , dt):
        self._trained_agent.play(self._board)

    
    def use_trained_agent(self, n):
        self._trained_agent = TrainedAgent(n)
        clock.schedule_interval(self.agent_play, 0.3)


    def game_over(self):
        self._board.reset_board()
        self.scorer.reset_game()
        self._score_label.text = 'Score: ' + '0'
        self._level_label.text = 'Level: ' + '1'
        print("Game has been reset")


    def do_score_update(self, lines):
        if lines == 0:
            return
        # clock.unschedule(self.do_board_update)

        self._score = self.scorer.update_score(lines)
        self._score_label.text = 'Score: ' + str(self._score)
        self._level_label.text = 'Level: ' + str(self.scorer.get_level())

        # clock.schedule_interval(self.do_board_update, self._interval)


    def on_draw(self):
        self.clear()
        self._score_label.draw()
        self._level_label.draw()
        array = self._board.get_board()
        for y in range(len(array)):
            for x in range(len(array[y])):
                if array[y][x] == 1:
                    self.Z.blit(x * 20, self.height - (y + 1) * 20)
                elif array[y][x] == 2:
                    self.L.blit(x * 20, self.height - (y + 1) * 20)
                elif array[y][x] == 3:
                    self.T.blit(x * 20, self.height - (y + 1) * 20)
                elif array[y][x] == 4:
                    self.O.blit(x * 20, self.height - (y + 1) * 20)
                elif array[y][x] == 5:
                    self.I.blit(x * 20, self.height - (y + 1) * 20)
        

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
            exit(0)
        if symbol == key.DOWN:
            rows_cleared = self._board.drop_block()
            if rows_cleared == -1:
                self.game_over()
                return
            self.do_score_update(rows_cleared)
        if symbol == key.SPACE:
            self._board.block_rotate()
        if symbol == key.RIGHT:
            self._board.block_right()
        if symbol == key.LEFT:
            self._board.block_left()


    def do_board_update(self, dt):
        rows_cleared = self._board.block_down()
        if rows_cleared == -1:
            self.game_over()
            return
        self.do_score_update(rows_cleared)
        
    
