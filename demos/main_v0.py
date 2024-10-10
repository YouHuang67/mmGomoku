import argparse

import torch
from mmengine.config import Config

import engine
import models
from projects.cppboard import Board
from mcts_search.wrappers import ActionValueEvaluatorV0, ActionValueEvaluatorV1
from mcts_search.trees import MCTSV0


def parse_position(input_str):
    try:
        row, col = map(int, input_str.split())
        return row, col
    except ValueError:
        raise ValueError("Invalid format! Please enter in 'row col' format.")

def main():
    parser = argparse.ArgumentParser(
        description='Start a game of Gomoku with AI interaction')
    parser.add_argument('-m', '--model-config',
                        type=str, required=True,
                        help='Path to the model configuration file')
    parser.add_argument('-w', '--weight-path',
                        type=str, required=True,
                        help='Path to the model weights file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the model on')
    parser.add_argument('-ave', '--action-value-evaluator',
                        type=str, default='ActionValueEvaluatorV1',
                        choices=['ActionValueEvaluatorV0',
                                 'ActionValueEvaluatorV1'],
                        help='Action-value evaluator')
    parser.add_argument('-vt', '--visit-times', type=int, default=200,
                        help='Number of MCTS simulations')
    parser.add_argument('--cpuct', type=float, default=1.0,
                        help='CPUCT constant')
    parser.add_argument('--max-vct-node-num', type=int, default=100_000,
                        help='Maximum number of nodes in the MCTS tree')
    parser.add_argument('--max-child-vct-node-num', type=int, default=10_000,
                        help='Maximum number of child nodes in the MCTS tree')
    args = parser.parse_args()

    cfg = Config.fromfile(args.model_config)
    model_config = {
        'backbone': cfg.model['backbone'],
        'neck': cfg.model['neck'],
        'decode_head': cfg.model['decode_head'],
        'device': args.device
    }

    evaluator_cls = {
        'ActionValueEvaluatorV0': ActionValueEvaluatorV0,
        'ActionValueEvaluatorV1': ActionValueEvaluatorV1
    }[args.action_value_evaluator]
    evaluator = evaluator_cls(**model_config)
    state_dict = torch.load(args.weight_path, map_location='cpu')['state_dict']
    state_dict.update({
        k: v for k, v in evaluator.state_dict().items()
        if not any(_ in k for _ in ['backbone', 'neck', 'decode_head'])
    })
    evaluator.load_state_dict(state_dict, strict=True)

    board = Board()
    mcts = MCTSV0(
        evaluator,
        [board],
        visit_times=args.visit_times,
        cpuct=args.cpuct,
        verbose=1,
        max_vct_node_num=args.max_vct_node_num,
        max_child_vct_node_num=args.max_child_vct_node_num
    )

    print("Welcome to the interactive Gomoku game!")
    mode = input("Select mode: 1 for AI first against Human, "
                 "2 for Human first against AI, 3 for AI vs AI: ")

    while mode not in ['1', '2', '3']:
        print("Invalid choice. Please choose again.")
        mode = input("Select mode: 1 for AI first against Human, "
                     "2 for Human first against AI, 3 for AI vs AI: ")

    if mode == '1':
        players = ['AI', 'Human']
    elif mode == '2':
        players = ['Human', 'AI']
    elif mode == '3':
        players = ['AI', 'AI']
    else:
        raise NotImplementedError(f'Mode {mode} not implemented')

    current_player = 0  # Start with player 1
    while not board.is_over:
        print(board)
        if players[current_player] == 'Human':
            valid_move = False
            while not valid_move:
                try:
                    user_input = input(f"Player {current_player + 1}, "
                                       f"enter your move (row col), "
                                       f"e.g., 7 7: ")
                    action = parse_position(user_input)
                    if board.is_legal(action):
                        board.move(action)
                        valid_move = True
                    else:
                        print("Illegal move! Try again.")
                except ValueError as e:
                    print(e)
        else:
            action, *_ = mcts.search(
                description=f"AlphaGomoku is thinking...")
            board.move(action)
            print(f"AlphaGomoku played {action}")
        mcts.move([action])

        current_player = 1 - current_player  # Switch player

    print("Game over!")
    print(board)


if __name__ == "__main__":
    main()
