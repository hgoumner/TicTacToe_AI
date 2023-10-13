"""
This file is used to train the Q model
"""

# --------------------------- #
# Import modules
# --------------------------- #

from TicTacToe import Game, QPlayer

# to save model
from pickle import dump

# --------------------------- #
# Global variables
# --------------------------- #

ITERATIONS = 2*(10**5)

# --------------------------- #
# Perform training
# --------------------------- #

epsilon = 0.9

p1 = QPlayer(symbol="X", eps=epsilon)
p2 = QPlayer(symbol="O", eps=epsilon)

game = Game(p1, p2)

winner = []
for iteration in range(ITERATIONS):
    game.play()
    winner.append(game.winner)
    game.reset()
    print(str(iteration) + '/' + str(ITERATIONS))

Q = game.Q

with open('winners.txt', 'w') as f:
    for line in winner:
        f.write(str(line) + '\n')

filename = "./models/QModel.model"
dump(Q, open(filename, "wb"))
