# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import os
import sys
from typing import Callable, Dict, List, Type, TypeVar

import pandas as pd
import torch
import torch.distributed


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class Meter(ABC):

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError(
            f'{Meter.reset.__name__} has not been implemented')

    @property
    def gatherable_inputs(self) -> Dict[str, Callable]:
        raise NotImplementedError(
            f'{Meter.gatherable_inputs.__qualname__} has not been implemented')

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError(
            f'{Meter.update.__name__} has not been implemented')


class ClassificationMeter(Meter):

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        super(ClassificationMeter, self).__init__()

    @property
    def top1_acc(self) -> float:
        return self._confusion_matrix.diag().sum() / self._confusion_matrix.sum()

    @property
    def top1_pc_acc(self) -> torch.Tensor:
        return self._confusion_matrix.diag() / self._confusion_matrix.sum(dim=1)

    def reset(self) -> None:
        self._confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), requires_grad=False, device=torch.device('cuda:0'))

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        for prediction, label in zip(predictions.view(-1), labels.view(-1)):
            self._confusion_matrix[prediction.long(), label.long()] += 1

    def tofile(self, dir: str, *, file_prefix: str = '', file_suffix: str = ''):

        if not os.path.exists(dir):
            os.makedirs(dir)

        save_path = os.path.join(
            dir, f'{file_prefix}confusion_matrix{file_suffix}.csv')
        with open(save_path, 'w') as f:
            df = pd.DataFrame(
                self._confusion_matrix.clone().detach().cpu().numpy())
            df.to_csv(f, sep=' ', header=False, index=False)


class FullVideoClassificationMeter(ClassificationMeter):

    def reset(self) -> None:
        super(FullVideoClassificationMeter, self).reset()
        self.confidences = dict()
        self.votes = dict()

    def update(self, confidences: torch.Tensor, labels: torch.Tensor, paths: List[str]):

        predictions = confidences.argmax(dim=1)
        super(FullVideoClassificationMeter, self).update(predictions, labels)

        for confidence, prediction, path in zip(confidences, predictions, paths):

            if not path in self.confidences:
                self.confidences[path] = torch.zeros_like(
                    confidence, requires_grad=False)
            self.confidences[path] += confidence

            if not path in self.votes:
                self.votes[path] = torch.zeros_like(
                    confidence, requires_grad=False)
            self.votes[path][prediction.long()] += 1

    def tofile(self, dir: str, *, file_prefix: str = '', file_suffix: str = '') -> None:
        super(FullVideoClassificationMeter, self).tofile(
            dir, file_prefix=file_prefix, file_suffix=file_suffix)

        confidences_path = os.path.join(
            dir, f'{file_prefix}confidences{file_suffix}.csv')
        with open(confidences_path, 'w') as f:
            confidences = [(path, *(confidence.clone().detach().cpu().numpy().tolist()), confidence.argmax().item())
                           for path, confidence in self.confidences.items()]
            df = pd.DataFrame(confidences)
            df.to_csv(f, sep=' ', header=False, index=False)

        votes_path = os.path.join(
            dir, f'{file_prefix}votes{file_suffix}.csv')
        with open(votes_path, 'w') as f:
            votes = [(path, *(vote.clone().detach().cpu().numpy().tolist()), vote.argmax().item())
                     for path, vote in self.votes.items()]
            df = pd.DataFrame(votes)
            df.to_csv(f, sep=' ', header=False, index=False)


class AverageMeter(Meter):
    """computes and stores the average and current value"""

    def __init__(self):
        super(AverageMeter, self).__init__()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def adamw_logger(optimizer):
    """ logging magnitude of first and second momentum buffers in adamw """
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get('state')
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))
    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}
