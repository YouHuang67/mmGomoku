import torch
import torch.nn as nn
from mmseg.models.builder import MODELS

from projects.cppboard import Board


class NearArea(nn.Module):

    def __init__(self, gap=3, size=Board.BOARD_SIZE):
        super(NearArea, self).__init__()
        self.size = size
        self.conv = nn.Conv2d(
            1, 1,
            kernel_size=(gap * 2 + 1),
            stride=1,
            padding=gap,
            bias=False)
        self.conv.weight.data.copy_(torch.ones_like(self.conv.weight))
        self.conv.weight.requires_grad_(False)

    def forward(self, legal_masks):
        masks = (~legal_masks.clone()).reshape(-1, 1, self.size, self.size)
        zero_masks = ~((~legal_masks).flatten(1).any(-1))
        if zero_masks.any():
            masks[zero_masks, 0, self.size // 2, self.size // 2] = True
        valid = self.conv(masks.float())
        return (valid > 0.0).flatten(1).bool()


class ActionValueEvaluatorV0(nn.Module):

    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 device='cpu',
                 gap=3,
                 size=Board.BOARD_SIZE):
        super(ActionValueEvaluatorV0, self).__init__()
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None
        self.device = device
        self.size = size
        self.near_area = NearArea(gap, size)

    @torch.no_grad()
    def forward(self, boards):
        """
        :param boards: a list of Board instances
        :return: a list of action probabilities (e.g. 15^2 float), and
                 a list of values
        """
        inputs = self.convert_board_to_tensors(boards, self.size, self.device)
        inputs, masks = inputs.float(), inputs[:, -1].bool().flatten(1)
        if self.neck is None:
            results = self.decode_head(self.backbone(inputs))
        else:
            results = self.decode_head(self.neck(self.backbone(inputs)))
        logits = results['action_probs'].reshape(-1, self.size ** 2)
        logits[~masks] = -torch.inf
        logits[~self.near_area(masks)] = -torch.inf
        action_probs = logits.softmax(dim=-1)
        values = results['values'].flatten().clip(-1.0, 1.0)
        return action_probs.detach().cpu().tolist(), \
               values.detach().cpu().tolist()

    @staticmethod
    def convert_board_to_tensors(boards, size, device):
        tensors = [torch.LongTensor(board.vector) for board in boards]
        tensors = torch.stack(tensors, dim=0).to(device)
        players = torch.LongTensor([board.player for board in boards])
        players = players.reshape(-1, 1).to(device)
        tensors = torch.stack([
            torch.eq(tensors, players).to(tensors),
            torch.eq(tensors, 1 - players).to(tensors),
            torch.eq(tensors, 2).to(tensors)
        ], dim=1)
        tensors = tensors.reshape(-1, 3, size, size)
        return tensors


if __name__ == '__main__':
    import models
    evaluator = ActionValueEvaluatorV0(
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
    print(evaluator(
        [
            Board(),
            Board([(7, 7), (7, 8)]),
            Board([(7, 7), (7, 8), (8, 7)])
        ]
    ))
