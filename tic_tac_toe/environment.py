import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action, player):
        if self.board[action] != 0:
            raise ValueError("Invalid move")
        self.board[action] = player
        if self.check_winner(player):
            self.done = True
            self.winner = player
        elif not self.available_actions():
            self.done = True
            self.winner = 0
        return self.board, self.done, self.winner

    def check_winner(self, player):
        for i in range(3):
            if all([self.board[i, j] == player for j in range(3)]) or all([self.board[j, i] == player for j in range(3)]):
                return True
        if all([self.board[i, i] == player for i in range(3)]) or all([self.board[i, 2 - i] == player for i in range(3)]):
            return True
        return False

    def render(self):
        for row in self.board:
            print(" | ".join(["X" if x == 1 else "O" if x == -1 else " " for x in row]))
            print("---------")
