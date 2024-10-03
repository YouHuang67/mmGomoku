from .board import BoardWrapper


__all__ = ['Board']


class Board(object):
    BOARD_SIZE = 15
    STONE_NUM = BOARD_SIZE ** 2
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    OPEN_FOUR = 1
    FOUR = 2
    OPEN_THREE = 3
    THREE = 4
    OPEN_TWO = 5

    def __init__(self, history=None, cppboard=None):
        if cppboard is None:
            self.cppboard = BoardWrapper()
        else:
            self.cppboard = cppboard
        self.history = []
        self.illegal_actions = set()
        if history is not None:
            for act in history:
                self.move(act)
                
    def is_legal(self, action):
        return action not in self.illegal_actions

    def move(self, action):
        if not self.is_legal(action):
            print(self)
            print('action: ', action)
            print('history: ', self.history)
            exit()
        self.cppboard.Move(self.action_flatten(*action))
        self.history.append(action)
        self.illegal_actions.add(action)
        return self

    def evaluate(self, max_node_num=100000):
        actions = self.cppboard.Evaluate(max_node_num)
        return [self.action_unflatten(act) for act in actions]

    def copy(self):
        return self.__class__(self.history)

    @property
    def attacker(self):
        return self.cppboard.Attacker()

    @property
    def is_over(self):
        return self.cppboard.IsOver()

    @property
    def player(self):
        return self.cppboard.Player()

    @property
    def winner(self):
        return self.cppboard.Winner()

    @property
    def key(self):
        key = 0
        for k in self.cppboard.Key():
            key = (key << 16) ^ k
        return key

    def next_key(self, action):
        key = 0
        for k in self.cppboard.NextKey(self.action_flatten(*action)):
            key = (key << 16) ^ k
        return key

    @property
    def vector(self):
        return self.cppboard.BoardVector()

    @staticmethod
    def get_homogenous_actions(action):
        action = Board.action_flatten(*action)
        actions = BoardWrapper.HomogenousActions(action)
        return [Board.action_unflatten(act) for act in actions]

    @staticmethod
    def action_flatten(row, col):
        return (row << 4) ^ col

    @staticmethod
    def action_unflatten(act):
        return act >> 4, act & 15

    def __repr__(self):
        players = {act: 'O' if i % 2 else 'X' 
                   for i, act in enumerate(self.history[:-1])}
        if len(self.history):
            players[self.history[-1]] = {0: '@', 1: '%'}[len(self.history) % 2]
        board_string = '  '
        for col in range(self.BOARD_SIZE):
            if col < 10:
                board_string += '  '
            else:
                board_string += ' 1'
        board_string += '\n  '
        for col in range(self.BOARD_SIZE):
            board_string += f' {col%10:d}'
        board_string += '\n'
        for row in range(self.BOARD_SIZE):
            board_string += f'{row:2d}'
            for col in range(self.BOARD_SIZE):
                board_string += ' '
                board_string += players.get((row, col), '_')
            board_string += '\n'
        return board_string

    __str__ = __repr__
