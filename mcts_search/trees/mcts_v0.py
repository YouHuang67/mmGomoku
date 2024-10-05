import time
import random
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.cppboard import Board
from mcts_search.nodes import NodeV0


class MCTSV0(object):

    def __init__(self,
                 evaluator,
                 boards,
                 visit_times=200,
                 cpuct=1.0,
                 verbose=1,
                 max_vct_node_num=100_000,
                 max_child_vct_node_num=10_000,
                 size=Board.BOARD_SIZE):

        self.evaluator = evaluator
        self.boards = boards
        self.visit_times = visit_times
        self.cpuct = cpuct
        self.verbose = verbose
        self.max_vct_node_num = max_vct_node_num
        self.max_child_vct_node_num = max_child_vct_node_num
        self.size = size

        self.roots = []
        self.node_tables = []
        for index, board in enumerate(boards):
            node_table = OrderedDict()
            root = NodeV0.get_node(board, node_table, cpuct=cpuct, index=index)
            self.roots.append(root)
            self.node_tables.append(node_table)

    def evaluate_actions_and_values(self, boards):
        probs_list = []
        value_list = []
        for board, probs, value in zip(boards, *self.evaluator(boards)):
            actions = board.evaluate(1)
            if len(actions) > 0:
                actions = list(map(self.flatten, actions))
            else:
                actions = [act for act, prob in enumerate(probs) if prob > 0.0]
            probs = {self.unflatten(act): probs[act] for act in actions}
            prob_sum = sum(probs.values())
            probs_list.append(
                {act: prob / prob_sum for act, prob in probs.items()})
            value_list.append(value)
        return probs_list, value_list

    def search(self, description=''):
        roots = []
        indices = []
        final_actions = []
        for index, root in enumerate(self.roots):
            board = root.board
            actions = board.copy().evaluate(self.max_vct_node_num)
            if len(actions) > 0 and board.attacker == board.player:
                final_actions.append(random.choice(actions))
            else:
                final_actions.append(None)
                roots.append(root)
                indices.append(index)

        if len(roots) == 0:
            return final_actions

        iterator = range(self.visit_times)
        if self.verbose:
            iterator = tqdm(iterator, description)
        for _ in iterator:
            nodes = []
            depths = []
            boards = []
            for root in roots:
                board = root.board.copy()
                node, depth = root.forward(board)

                if root not in list(node.parents.values()) + [node]:
                    max_node_num = self.max_child_vct_node_num
                else:
                    max_node_num = self.max_vct_node_num
                if node.is_leaf or node.evaluate_leaf(board.copy(), max_node_num):
                    node.backward(node.leaf_value, depth)
                    continue

                nodes.append(node)
                depths.append(depth)
                boards.append(board)

            if len(boards) == 0:
                continue

            probs_list, value_list = self.evaluate_actions_and_values(boards)

            for node, probs, board, value, depth in zip(
                    nodes, probs_list, boards, value_list, depths
            ):
                node.expand(probs, board, self.node_tables[node.index])
                node.backward(value, depth)

        for index, root in zip(indices, roots):
            children = list(root.children.items())
            if len(children) > 0:
                random.shuffle(children)
                final_actions[index] = sorted(
                    children, key=lambda x: x[1].visit
                )[-1][0]
            else:
                final_actions[index] = root.attack_action
        return final_actions

    def move(self, actions):
        roots = self.roots
        self.roots = []
        for root, action in list(zip(roots, actions)):
            index = root.index
            board = root.board.copy().move(action)
            if board.is_over:
                continue
            root = NodeV0.get_node(
                board, self.node_tables[index], cpuct=self.cpuct, index=index)
            root.board = board
            self.roots.append(root)

    def flatten(self, *args):
        row, col = (args[0] if len(args) == 1 else args)
        return row * self.size + col

    def unflatten(self, action):
        return action // self.size, action % self.size


if __name__ == '__main__':
    import models
    from mcts_search.wrappers import ActionValueEvaluatorV0

    _board = Board()

    _evaluator = ActionValueEvaluatorV0(
        backbone=dict(
            type='SimpleResNet',
            depth=16,
            channels=64,
            kernel_size=3),
        neck=None,
        decode_head=dict(
            type='SimplePolicyValueHead',
            depth=3,
            channels=64,
            kernel_size=3),
        device='cpu'
    )
    _evaluator.load_state_dict(torch.load(
        'work_dirs/cnn2410/vct_actions/'
        '241004_pvt_v0_resnet_plain_d16_c64_k3_simaug_vct_actions_bs16_40k/'
        'iter_40000.pth', map_location='cpu'
    )['state_dict'], strict=False)
    mcts = MCTSV0(_evaluator, [_board])
    while not _board.is_over:
        action, *_ = mcts.search()
        _board.move(action)
        mcts.move([action])
        print(_board)
