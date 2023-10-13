"""
This file is used to create a game between a human and a Q player
"""

# --------------------------- #
# Import modules
# --------------------------- #

from TicTacToe import Game, Human, QPlayer

# --------------------------- #
# Run game
# --------------------------- #

p1 = Human(symbol='X')
p2 = QPlayer(symbol='O')

game = Game(p1, p2)

game.play()