import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import mmengine
from mmengine.config import Config

import engine
import models
from projects.cppboard import Board
from mcts_search.wrappers import (ActionValueEvaluatorV0,
                                  ActionValueEvaluatorV0Rand,
                                  ActionValueEvaluatorV1)
from mcts_search.trees import MCTSV0


def get_random_actions(step, gap=3):
    if step == 0:
        return []
    size = Board.BOARD_SIZE
    actions = torch.rand(size ** 2).argsort().numpy().tolist()
    random_actions = [(np.random.randint(size // 2 - gap, size // 2 + gap + 1),
                       np.random.randint(size // 2 - gap, size // 2 + gap + 1))]
    for action in actions:
        if len(random_actions) == step:
            break
        row, col = action // size, action % size
        if any([(row == r and col == c) for r, c in random_actions]):
            continue
        if any([abs(row - r) <= gap and abs(col - c) <= gap
                for r, c in random_actions]):
            random_actions.append((row, col))
    return random_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-config', type=str, required=True)
    parser.add_argument('-w', '--weight-path', type=str, required=True)
    parser.add_argument('-ave', '--action-value-evaluator',
                        type=str, default='ActionValueEvaluatorV0Rand',
                        choices=['ActionValueEvaluatorV0',
                                 'ActionValueEvaluatorV0Rand',
                                 'ActionValueEvaluatorV1'])
    parser.add_argument('-n', '--num-boards', type=int, default=1000)
    parser.add_argument('-vt', '--visit-times', type=int, default=200)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--max-vct-node-num', type=int, default=100_000)
    parser.add_argument('--max-child-vct-node-num', type=int, default=10_000)
    parser.add_argument('-bs', '--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('-mrs', '--max-rand-step', type=int, default=5)
    parser.add_argument('--gap', type=int, default=3)
    args = parser.parse_args()

    work_dir = Path('work_dirs/data_generation')
    work_dir = work_dir / Path(__file__).stem
    work_dir = work_dir / f'{time.strftime("%Y%m%d_%H%M%S")}'
    work_dir.mkdir(parents=True, exist_ok=True)
    mmengine.dump(args.__dict__, work_dir / 'config.json', indent=4)

    model_cfg = {
        k: v for k, v in Config.fromfile(args.model_config).model.items()
        if k in ['backbone', 'neck', 'decode_head']}
    evaluator = {
        'ActionValueEvaluatorV0': ActionValueEvaluatorV0,
        'ActionValueEvaluatorV0Rand': ActionValueEvaluatorV0Rand,
        'ActionValueEvaluatorV1': ActionValueEvaluatorV1
    }[args.action_value_evaluator](**model_cfg, device=args.device)
    evaluator.load_state_dict(
        torch.load(args.weight_path, map_location='cpu')['state_dict'],
        strict=False)
    evaluator = evaluator.to(args.device)
    evaluator.eval()

    count = 0
    progress_bar = tqdm(range(args.num_boards), ncols=100)
    while count < args.num_boards:
        start = count
        batch_size = min(args.batch_size, args.num_boards - count)
        boards = [Board() for _ in range(batch_size)]
        mcts = MCTSV0(
            evaluator,
            boards,
            visit_times=args.visit_times,
            cpuct=args.cpuct,
            max_vct_node_num=args.max_vct_node_num,
            max_child_vct_node_num=args.max_child_vct_node_num,
            verbose=0
        )
        step = np.random.randint(args.max_rand_step + 1)
        actions_list = list(zip(*[
            get_random_actions(step, args.gap) for _ in boards
        ]))
        for actions in actions_list:
            for board, action in zip(boards, actions):
                board.move(action)
            mcts.move(actions)
        _boards = boards
        while len(_boards) > 0:
            actions = mcts.search()
            new_boards = []
            for board, action in zip(_boards, actions):
                board.move(action)
                if not board.is_over:
                    new_boards.append(board)
                else:
                    count += 1
                    progress_bar.update()
            mcts.move(actions)
            _boards = new_boards
        for board in boards:
            mmengine.dump(
                dict(history=board.history),
                work_dir / f'{start:05d}.json')
            start += 1


if __name__ == '__main__':
    main()
