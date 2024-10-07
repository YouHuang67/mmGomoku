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
from mcts_search.wrappers import ActionValueEvaluatorV0, ActionValueEvaluatorV1
from mcts_search.trees import MCTSV0


def perturb_probabilities_dirichlet(probs, alpha_scale, eps=1e-8):
    probs_np = np.array(probs)

    nonzero_indices = probs_np > 0

    alpha = probs_np[nonzero_indices] * alpha_scale
    alpha = np.maximum(alpha, eps)

    perturbed_nonzero_probs = np.random.dirichlet(alpha)
    perturbed_nonzero_probs = np.maximum(perturbed_nonzero_probs, eps)
    perturbed_nonzero_probs /= np.sum(perturbed_nonzero_probs)

    perturbed_probs = np.zeros_like(probs_np)
    perturbed_probs[nonzero_indices] = perturbed_nonzero_probs

    return perturbed_probs.tolist()


def wrap_evaluator(evaluator, alpha_scale):
    def eval_func(boards):
        probs, values = evaluator(boards)
        probs = [perturb_probabilities_dirichlet(prob, alpha_scale)
                 for prob in probs]
        return probs, values
    return eval_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-config', type=str, required=True)
    parser.add_argument('-w', '--weight-path', type=str, required=True)
    parser.add_argument('-ave', '--action-value-evaluator',
                        type=str, default='ActionValueEvaluatorV0',
                        choices=['ActionValueEvaluatorV0',
                                 'ActionValueEvaluatorV1'])
    parser.add_argument('-n', '--num-boards', type=int, default=1000)
    parser.add_argument('-vt', '--visit-times', type=int, default=200)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--max-vct-node-num', type=int, default=100_000)
    parser.add_argument('--max-child-vct-node-num', type=int, default=10_000)
    parser.add_argument('-as', '--alpha-scale', type=float, default=0.1)
    parser.add_argument('-bs', '--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
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
        'ActionValueEvaluatorV1': ActionValueEvaluatorV1
    }[args.action_value_evaluator](**model_cfg, device=args.device)
    evaluator.load_state_dict(
        torch.load(args.weight_path, map_location='cpu')['state_dict'],
        strict=False)
    evaluator = evaluator.to(args.device)
    evaluator.eval()
    evaluator = wrap_evaluator(evaluator, args.alpha_scale)

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
