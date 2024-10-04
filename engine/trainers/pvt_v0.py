import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import MODELS

from .base import BaseTrainer


def topk_accuracy(output, target, topk=(1, 2, 3)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, -1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@MODELS.register_module()
class PolicyValueTrainerV0(BaseTrainer):

    def forward_train(self, inputs, targets, data_samples, **kwargs):
        """
        :param inputs: (B, 3, H, W)
        :param targets: (B, )
        :param data_samples:
        :return:
        """
        inputs = torch.cat([
            inputs,
            torch.stack([inputs[:, 1], inputs[:, 0], inputs[:, 2]], dim=1)
        ], dim=0)
        if self.neck is None:
            results = self.decode_head(self.backbone(inputs))
        else:
            results = self.decode_head(self.neck(self.backbone(inputs)))
        action_probs = results['action_probs'].chunk(2, dim=0)[0]  # B, H, W
        action_probs = action_probs.flatten(1)  # B, H * W
        values = results['values']  # 2 * B

        losses = dict()
        losses.update(dict(action_loss=F.cross_entropy(action_probs, targets)))
        top1, top2, top3 = topk_accuracy(action_probs, targets, topk=(1, 2, 3))
        losses.update(dict(top1=top1, top2=top2, top3=top3))

        targets = torch.ones_like(values)
        targets[len(targets) // 2:] = -1
        losses.update(dict(value_loss=F.mse_loss(values, targets)))
        losses.update(dict(abs=F.l1_loss(values, targets)))
        return losses

    @torch.no_grad()
    def forward_test(self, inputs, targets, data_samples, **kwargs):
        raise NotImplementedError
