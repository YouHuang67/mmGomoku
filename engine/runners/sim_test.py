import numpy as np
import torch
import torch.nn.functional as F
import mmengine
from mmengine import print_log
from mmengine.registry import LOOPS

from ..trainers.pvt_v0 import topk_accuracy
from .base import BaseTestLoop


@LOOPS.register_module()
class SimpleTestLoop(BaseTestLoop):

    def __init__(self,
                 runner,
                 dataloader,
                 evaluator=None,
                 topk=(1, 2, 3)):
        super(SimpleTestLoop, self).__init__(runner, dataloader, evaluator)
        self.topk = topk

    def process(self, outputs, data_batch):
        action_probs, values, targets = outputs  # assume batch size is 1
        assert len(action_probs) == 1
        return list(map(lambda x: x.item(),
                        topk_accuracy(action_probs, targets, topk=self.topk))), \
               F.l1_loss(values, torch.ones_like(values)).item()

    def compute_metrics(self, results):
        topk_acc, value_loss = list(zip(*results))
        topk_acc_list = list(zip(*topk_acc))
        results = dict()
        metrics = dict()
        for idx, k in enumerate(self.topk):
            results[f'top{k}'] = topk_acc_list[idx]
            metrics[f'top{k}'] = float(np.mean(topk_acc_list[idx]))
            print_log(f'top{k}: {metrics[f"top{k}"]:.2f}%', logger='current')
        results['value_loss'] = value_loss
        metrics['value_loss'] = float(np.mean(value_loss))
        print_log(f'value_loss: {metrics["value_loss"]:.2f}', logger='current')

        work_dir = self.runner.log_dir
        mmengine.dump(results,
                      f'{work_dir}/simple_test_res.json', indent=4)
        mmengine.dump(metrics,
                      f'{work_dir}/simple_test_met.json', indent=4)
