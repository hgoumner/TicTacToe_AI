"""
This file contains the TicTacToe game
"""

# --------------------------- #
# Import modules
# --------------------------- #

# import sys
import copy
from random import choice
import numpy as np

# --------------------------- #
# Global variables
# --------------------------- #
GRID_EDGE_LENGTH = 3
Q_REWARD = 1
Q_DEFAULT = 1.0

# --------------------------- #
# class definitions
# --------------------------- #

class Board(object):
    """
    Board definition
    """

    def __init__(self, grid=np.ones((GRID_EDGE_LENGTH, GRID_EDGE_LENGTH))*np.nan):
        self.grid = grid

    def get_openfields(self):
        """
        Get the unused fields
        """

        return [(i, j) for i in range(GRID_EDGE_LENGTH) for j in range(GRID_EDGE_LENGTH) if np.isnan(self.grid[i][j])]

    @staticmethod
    def tuple_to_int(grid_index):
        """
        Convert field tuple to integer value representing field
        """

        nrows = GRID_EDGE_LENGTH

        field_number = grid_index[0]*nrows + grid_index[1]

        return field_number

    @staticmethod
    def int_to_tuple(field_number):
        """
        Convert integer value representing field to field tuple
        """

        row = field_number // GRID_EDGE_LENGTH
        col = field_number % GRID_EDGE_LENGTH

        return (row, col)

    @staticmethod
    def encode_symbol(symbol):
        """
        encode the players symbol 0 or 1
        """

        if symbol == 'X':
            return 1
        else:
            return 0

    @staticmethod
    def decode_symbol(value):
        """
        decode the players symbol 0 or 1
        """

        if value == 1:
            return 'X'
        elif value == 0:
            return 'O'
        else:
            return '-'

    def set_symbol_on_grid(self, field, symbol):
        """
        update board with respective value
        """

        current = Board.encode_symbol(symbol)
        self.grid[tuple(field)] = current

    def show_grid(self):
        """
        Show the current status of the grid
        """

        divider = '---+---+---\n'
        result = '\n '
        result += ' | '.join([str(self.decode_symbol(value)) for value in self.grid[0][:]]) + '\n'
        for row in range(1, self.grid.shape[0]):
            result += divider +  ' ' + ' | '.join([str(self.decode_symbol(value)) for value in self.grid[row][:]]) + '\n'

        print(result)

    def get_winner(self):
        """
        get winner
        """

        # rows
        rows = [self.grid[i, :] for i in range(GRID_EDGE_LENGTH)]

        # columns
        cols = [self.grid[:, i] for i in range(GRID_EDGE_LENGTH)]

        # diagonals
        diag = [np.array([self.grid[i, i] for i in range(GRID_EDGE_LENGTH)])]
        cross_diag = [np.array([self.grid[2 - i, i] for i in range(GRID_EDGE_LENGTH)])]

        # check if any of the winning options is satisfied
        win_option = np.concatenate((rows, cols, diag, cross_diag))
        any_option = lambda x: any([np.array_equal(option, x) for option in win_option])
        if any_option(np.ones(GRID_EDGE_LENGTH)):
            # player one wins
            return "X"
        elif any_option(np.zeros(GRID_EDGE_LENGTH)):
            # player two wins
            return "O"

    def check_status(self):
        """
        check game status
        """

        # is there a winner or are all fields filled up - tie?
        return (self.get_winner() is not None) or (not np.any(np.isnan(self.grid)))

    ########################
    #### For Q-Learning ####
    ########################

    def create_new_board(self, field, symbol):
        """
        Create a new board with the current state
        """

        # make copy of board and set the respective symbol on the grid
        new_board = copy.deepcopy(self)
        new_board.set_symbol_on_grid(field, symbol)

        return new_board

    def encode_grid_state(self, symbol):
        """
        create a string representation of the state of the board and turn
        """

        fill_value = 9
        new_grid = copy.deepcopy(self.grid)
        np.place(new_grid, np.isnan(new_grid), fill_value)

        return "".join(map(str, (list(map(int, new_grid.flatten()))))) + symbol

    def give_reward(self):
        """
        Assign rewards for win, loss, and tie
        """

        if self.check_status():
            if self.get_winner() is not None:
                if self.get_winner() == "X":
                    # Player one won - positive reward
                    return Q_REWARD
                elif self.get_winner() == "O":
                    # Player two won - a negative reward
                    return -1*Q_REWARD
            else:
                # Tie - smaller reward
                return Q_REWARD/2
        else:
            # game is still ongoing - no reward
            return 0.0

    ########################
    ##### For MiniMax ######
    ########################

    def evaluate(self):
        """
        Feedback for MiniMax algorithm - like Q learning rewards
        """

        if self.get_winner() is not None:
            if self.get_winner() == "X":
                # Player one won - positive reward
                return 1
            elif self.get_winner() == "O":
                # Player two won - a negative reward
                return -1
        else:
            # Tie - smaller reward
            return 0

class BasePlayer(object):
    """
    Base Player definition
    """

    def __init__(self, symbol):
        self.symbol = symbol

class Human(BasePlayer):
    """
    Human player definition - interactive
    """

    def __init__(self, symbol):
        super(Human, self).__init__(symbol=symbol)

    @staticmethod
    def get_field(board):
        """
        Determine next field to be selected
        """

        fields = board.get_openfields()
        if fields:
            while True:
                selection = input(f"Please select a valid field ({', '.join([str(board.tuple_to_int(i)) for i in fields])}): ")
                if board.int_to_tuple(int(selection)) in fields:
                    break
                else:
                    print('Wrong input. Try again!')

        return board.int_to_tuple(int(selection))

class Random(BasePlayer):
    """
    Random player definition
    """

    def __init__(self, symbol):
        super(Random, self).__init__(symbol=symbol)

    @staticmethod
    def get_field(board):
        """
        Determine next field to be selected
        """

        fields = board.get_openfields()
        if fields:
            return choice(fields)

# TODO: needs to be fixed because the field selection is faulty
class MinMax(BasePlayer):
    """
    MinMax player definition
    """

    def __init__(self, symbol):
        super(MinMax, self).__init__(symbol=symbol)

    @staticmethod
    def minimax(board, depth, player):
        """
        Minimax algorithm for tic tac toe
        """

        if player == -1:
            best = [-1, -1, -np.inf]
        else:
            best = [-1, -1, np.inf]

        if depth == 0 or board.check_status():
            score = board.evaluate()
            return [-1, -1, score]

        for cell in board.get_openfields():
            x, y = cell[0], cell[1]
            board.grid[x][y] = player
            score = MinMax.minimax(board, depth - 1, -player)
            board.grid[x][y] = 0
            score[0] = x
            score[1] = y

            if player == -1:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        return best

    @staticmethod
    def get_field(board, game):
        """
        Determine the next field to be selected
        """

        # depth of game tree
        depth = len(board.get_openfields())
        if depth == 0 or board.check_status():
            return

        # choose random start or minimax output
        if depth == 9:
            x = choice([0, 1, 2])
            y = choice([0, 1, 2])
        else:
            new_board = copy.deepcopy(board)
            res = MinMax.minimax(new_board, depth, 1)
            x, y = res[0], res[1]

        return x, y

class QPlayer(BasePlayer):
    """
    Q player definition
    """

    def __init__(self, symbol, Q=dict(), eps=0.2):
        super(QPlayer, self).__init__(symbol=symbol)
        self.Q   = Q
        self.eps = eps

    def get_field(self, board):
        """
        Determine the next field to be selected
        """

        # Select a field at random ("epsilon-greedy" exploration) with probability eps
        if np.random.uniform() < self.eps:
            return Random.get_field(board)
        else:
            key = QPlayer.create_key(board, self.symbol, self.Q)
            Q_cur = self.Q[key]

            if self.symbol == "X":
                return QPlayer.stochastic_minmax(Q_cur, max)
            elif self.symbol == "O":
                return QPlayer.stochastic_minmax(Q_cur, min)

    @staticmethod
    def create_key(board, symbol, Q):
        """
        Create key for the current state of the board and turn and append it to Q, if necessary
        """

        key = board.encode_grid_state(symbol)

        # if key is not found, append it
        if Q.get(key) is None:
            fields = board.get_openfields()
            Q[key] = {field: Q_DEFAULT for field in fields}

        return key

    @staticmethod
    def stochastic_minmax(Q_vals, minmax):
        """
        Choose either min or max if there are ties based on random choice
        """

        Q_minmax = minmax(list(Q_vals.values()))

        # If there is more than one field corresponding to the maximum Q-value,
        # choose one at random
        if list(Q_vals.values()).count(Q_minmax) > 1:
            best_options = [f for f in list(Q_vals.keys()) if Q_vals[f] == Q_minmax]
            field = best_options[np.random.choice(len(best_options))]
        else:
            field = minmax(Q_vals, key=Q_vals.get)

        return field

class Game(object):
    """
    Game rules definition
    """

    def __init__(self, player_one, player_two, Q_learn=None, Q=dict(), alpha=0.3, gamma=0.9):
        # players
        self.player_one = player_one
        self.player_two = player_two
        self.player_cur = player_one
        self.player_oth = player_two
        self.winner = 0

        # Board
        self.board = Board()

        # Q-Learning
        self.Q_learn = Q_learn
        if self.Q_learn:
            self.Q = Q

            # learning rate
            self.alpha = alpha

            # discount rate
            self.gamma = gamma

            self.Q_share()

    def _print_message(self, message):
        """
        print message
        """

        print()
        print("-" * 100)
        print(message)
        print("-" * 100)
        print()

    def create_game(self):
        """
        create the game
        """

        if isinstance(self.player_one, Human) or isinstance(self.player_two, Human):
            self._print_message("Welcome to a game of tic tac toe")
            print("Select a starting field (0-8)\n")
            print(" 0 | 1 | 2 \n---+---+---\n 3 | 4 | 5 \n---+---+---\n 6 | 7 | 8 \n")

    def switch_players(self):
        """
        Switch the current player
        """

        if self.player_cur == self.player_one:
            self.player_cur = self.player_two
            self.player_oth = self.player_one
        else:
            self.player_cur = self.player_one
            self.player_oth = self.player_two

    def perform_turn(self, field):
        """
        Perform turn
        """

        if self.Q_learn:
            self.learn_Q(field)
        self.board.set_symbol_on_grid(field, self.player_cur.symbol)

        # print grid if human is involved
        if isinstance(self.player_one, Human) or isinstance(self.player_two, Human):
            self.board.show_grid()
        if self.board.check_status():
            if isinstance(self.player_one, Human) or isinstance(self.player_two, Human):
                self.print_result()
        else:
            self.switch_players()

    def player_turn(self):
        """
        Get the selection and perform the move
        """

        if not isinstance(self.player_cur, MinMax):
            field = self.player_cur.get_field(self.board)
        else:
            field = self.player_cur.get_field(self.board, self)

        self.perform_turn(field)

    def play(self):
        """
        Play the game
        """

        # opening message
        self.create_game()

        if isinstance(self.player_one, QPlayer) and isinstance(self.player_two, Human):
            initial_selection = self.player_one.get_field(self.board)
            self.perform_turn(initial_selection)
        else:
            while not self.board.check_status():
                self.player_turn()

        if self.board.get_winner() is None:
            self.winner = 0
        else:
            if self.player_cur == self.player_one:
                self.winner = 1
            else:
                self.winner = 2

    def print_result(self):
        """
        Print the result of the game
        """

        if self.board.get_winner() is None:
            self._print_message("The game ends in a tie.")
        else:
            if self.player_cur == self.player_one:
                self._print_message("Player one has won the game!")
            else:
                self._print_message("Player two has won the game!")

    def reset(self):
        """
        Reset the game
        """

        if isinstance(self.player_one, Human) or isinstance(self.player_two, Human):
            self._print_message("Starting a new game now")

        # create new board
        self.board = Board(grid=np.ones((GRID_EDGE_LENGTH, GRID_EDGE_LENGTH))*np.nan)

        # assign players
        self.player_cur = self.player_one
        self.player_oth = self.player_two

        # play
        self.play()

    ########################
    #### For Q-Learning ####
    ########################

    @property
    def Q_learn(self):
        """
        Determine if Q learning is to be activated
        """

        if self._Q_learn is not None:
            return self._Q_learn
        if isinstance(self.player_one, QPlayer) or isinstance(self.player_two, QPlayer):
            return True

    @Q_learn.setter
    def Q_learn(self, _Q_learn):
        self._Q_learn = _Q_learn

    def Q_share(self):
        """
        # Q is shared with the QPlayers to help them make their field selections
        """

        if isinstance(self.player_one, QPlayer):
            self.player_one.Q = self.Q
        if isinstance(self.player_two, QPlayer):
            self.player_two.Q = self.Q

    def learn_Q(self, field):
        """
        This should be called before setting the symbol and after receiving a move from an instance of Player
        """

        key = QPlayer.create_key(self.board, self.player_cur.symbol, self.Q)
        new_board = self.board.create_new_board(field, self.player_cur.symbol)
        reward = new_board.give_reward()
        new_key = QPlayer.create_key(new_board, self.player_oth.symbol, self.Q)
        if new_board.check_status():
            expectation = reward
        else:
            # The Q values represent the expected future reward for player X for each available move in the next state
            # (after the move has been made)
            Qs_new = self.Q[new_key]
            if self.player_cur.symbol == 'X':
                # Current player is player one - choose minimum Q value
                expectation = reward + (self.gamma * min(Qs_new.values()))
            elif self.player_cur.symbol == 'O':
                # Current player is player two - choose maximum Q value
                expectation = reward + (self.gamma * max(Qs_new.values()))

        change = self.alpha * (expectation - self.Q[key][field])
        self.Q[key][field] += change
