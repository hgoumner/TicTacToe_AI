"""
This file is used to create a game between two human players
"""

# --------------------------- #
# Import modules
# --------------------------- #

from TicTacToe import Game, Human

# --------------------------- #
# Run game
# --------------------------- #

p1 = Human(symbol='X')
p2 = Human(symbol='O')

game = Game(p1, p2)

game.play()