"""
This file is used to create a game between a human and a trained Q player
"""

# --------------------------- #
# Import modules
# --------------------------- #

from TicTacToe import Game, Human, QPlayer

from pickle import load

# --------------------------- #
# Run game
# --------------------------- #

Q = load(open('./models/QModel.model', "rb"))

p1 = Human(symbol='X')
p2 = QPlayer(symbol='O', eps=0)

game = Game(p1, p2, Q=Q)

game.play()