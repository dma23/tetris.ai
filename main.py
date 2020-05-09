import pyglet
from pyglet.window import key
from model.agent import Agent
from game.view import View
import numpy as np
import sys

# define constants
_width, _height = 300, 400


if __name__ == '__main__':
    _screen = View(_width, _height, 'Tetris')
    _board = _screen._board

    if len(sys.argv) == 1:
        _agent = Agent()
        _agent.run(_board)
    else:
        model = sys.argv[1]
        _screen.use_trained_agent(model)


    pyglet.app.run()
