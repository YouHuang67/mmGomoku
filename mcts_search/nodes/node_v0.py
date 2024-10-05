import random


class NodeV0(object):

    def __init__(self,
                 prob,
                 key,
                 player,
                 level,
                 cpuct=1.0,
                 board=None,
                 **kwargs):
        self._prob = prob
        self._prob_n = 1
        self.key = key
        self.player = player
        self.level = level
        self.cpuct = cpuct
        self.board = board
        self.extra_kwargs = kwargs
        self.__dict__.update(kwargs)

        self.visit = 0
        self.value = 0.0
        self.children = dict()
        self.parents = dict()
        self.is_expanded = False
        self.leaf_value = None
        self.attack_action = None

    def evaluate_leaf(self, board, max_node_num=100000):
        assert len(self.children) == 0
        if board.is_over:
            if board.winner == 1 - board.player:
                self.leaf_value = -1.0
            else:
                self.leaf_value = 0.0
            return True
        else:
            actions = board.evaluate(max_node_num)
            if board.attacker == board.player:
                self.leaf_value = 1.0
                self.attack_action = actions[0]
                return True
            return False

    def expand(self, probs, board, node_table):
        assert not self.is_expanded
        for action, prob in probs.items():
            self.children[action] = \
                self.get_node(
                    board, node_table, action,
                    prob, self.cpuct, **self.extra_kwargs)
        self.is_expanded = True

    def forward(self, board, parent=None, depth=0):
        self.visit += 1
        if parent is not None:
            self.parents[parent.key] = parent
        if not self.is_expanded:
            return self, depth
        scores = self.get_scores()
        best_action, best_score = random.choice(list(scores.items()))
        for action, score in scores.items():
            if score > best_score:
                best_action = action
                best_score = score
        return self.children[best_action].forward(
            board.move(best_action), self, depth + 1)

    def backward(self, value, depth):
        self.value += value
        if depth:
            for key in list(self.parents.keys()):
                self.parents.pop(key).backward(-value, depth - 1)

    @property
    def is_leaf(self):
        return self.leaf_value is not None

    @property
    def prob(self):
        return self._prob / self._prob_n

    @property
    def score(self):
        return self.value / self.visit if self.visit else 0.0

    def visit_bonus(self, total_visit):
        return self.prob * total_visit ** 0.5 / (1 + self.visit)

    def get_scores(self):
        total_visit = sum(node.visit for node in self.children.values()) + 1
        return {action: -node.score + node.cpuct * node.visit_bonus(total_visit)
                for action, node in self.children.items()}

    def update_prob(self, prob):
        self._prob += prob
        self._prob_n += 1
        return self

    def detach(self):
        for child in self.children.values():
            child.parents.pop(self.key, None)
        return self.key

    @classmethod
    def get_node(cls,
                 board,
                 node_table,
                 action=None,
                 prob=1.0,
                 cpuct=1.0,
                 **kwargs):
        if action is None:
            key = board.key
            player = board.player
            level = len(board.history)
            board = board.copy()
        else:
            key = board.next_key(action)
            player = 1 - board.player
            level = len(board.history) + 1
            board = None
        return node_table.setdefault(
            key,
            cls(prob, key, player, level, cpuct, board, **kwargs)
        )
