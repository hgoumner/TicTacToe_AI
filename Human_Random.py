"""
This file is used to create a game between a human and a Random player
"""

# --------------------------- #
# Import modules
# --------------------------- #

from TicTacToe import Game, Human, Random

# --------------------------- #
# Run game
# --------------------------- #

p1 = Human(symbol='X')
p2 = Random(symbol='O')

game = Game(p1, p2)

game.play()