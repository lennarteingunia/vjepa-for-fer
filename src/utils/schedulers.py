# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import wraps
import math
import warnings
import weakref

import torch
from torch.optim.optimizer import Optimizer


# class WarmupCosineSchedule(object):

#     def __init__(
#         self,
#         optimizer,
#         warmup_steps,
#         start_lr,
#         ref_lr,
#         T_max,
#         last_epoch=-1,
#         final_lr=0.
#     ):
#         self.optimizer = optimizer
#         self.start_lr = start_lr
#         self.ref_lr = ref_lr
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.T_max = T_max - warmup_steps
#         self._step = 0.

#     def step(self):
#         self._step += 1
#         if self._step < self.warmup_steps:
#             progress = float(self._step) / float(max(1, self.warmup_steps))
#             new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
#         else:
#             # -- progress after warmup
#             progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
#             new_lr = max(self.final_lr,
#                          self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

#         for group in self.optimizer.param_groups:
#             group['lr'] = new_lr

#         return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd


class WarmupCosineLRSchedule(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        T_max: int,
        final_lr: float = 0.0,
    ):
        assert warmup_steps <= T_max
        assert start_lr >= 0
        assert ref_lr >= 0
        assert final_lr >= 0

        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.T_max = T_max
        self.final_lr = final_lr
        super(WarmupCosineLRSchedule, self).__init__(optimizer)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            progress = float(self._step_count) / \
                float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step_count -
                             self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        return [new_lr for _ in self.optimizer.param_groups]


class WDScheduler:

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1
    ) -> None:

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_wd', group['weight_decay'])
                group.setdefault('WD_exclude', False)
        else:
            for group_idx, group in enumerate(optimizer.param_groups):
                if 'initial_wd' not in group:
                    raise KeyError(
                        f'param "initial_lr" is not specified in param_groups[{group_idx}] when resuming an optimizer')
                if 'WD_exclude' not in group:
                    raise KeyError(
                        f'param "WD_exclude" is not specified in param_groups[{group_idx}] when resuming an optimizer')
        self.base_wd = [group['initial_wd']
                        for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                return method

            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__

            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self._initial_step()

    def _initial_step(self):
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_wd(self):
        return self._last_wd

    def print_wd(self, is_verbose: bool, group, wd, epoch=None):
        if is_verbose:
            if epoch is None:
                print(f'Adjusting weight decay of group {group} to {wd:.4e}')
            else:
                epoch_str = f'{epoch:.2f}' if isinstance(
                    epoch, float) else f'{epoch:.5d}'
                print(
                    f'Epoch {epoch_str}: adjusting weight decay of group to {wd:.4e}')

    def step(self):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overriden after weight decay scheduler initialization. Please make sure to call `optimizer.step()` before `wd_scheduler.step()`.", UserWarning)
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `wd_scheduler.step()` before `optimizer.step()`. Please call `optimizer.step()` first. Failure to do so will result in skipping of the first value of the weight decay schedule.", UserWarning)

        self._step_count += 1
        with _enable_get_wd_call(self):

            self.last_epoch += 1
            values = self.get_wd()

            for group_idx, (group, wd) in enumerate(zip(self.optimizer.param_groups, values)):
                if not group['WD_exclude']:
                    group['weight_decay'] = wd

            self._last_wd = [group['weight_decay']
                             for group in self.optimizer.param_groups]

    def get_wd(self):
        raise NotImplementedError()


class LRWDSchedule(torch.optim.lr_scheduler.LRScheduler, WDScheduler):

    def __init__(
        self,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        wd_scheduler: WDScheduler
    ) -> None:
        assert lr_scheduler.optimizer is wd_scheduler.optimizer
        assert lr_scheduler.last_epoch == wd_scheduler.last_epoch

        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        torch.optim.lr_scheduler.LRScheduler.__init__(
            self, optimizer=lr_scheduler.optimizer, last_epoch=lr_scheduler.last_epoch)
        WDScheduler.__init__(
            self, optimizer=wd_scheduler.optimizer, last_epoch=wd_scheduler.last_epoch)

    def get_lr(self) -> float:
        return self.lr_scheduler.get_lr()

    def get_wd(self):
        return self.wd_scheduler.get_wd()

    def step(self):
        self.lr_scheduler.step()
        self.wd_scheduler.step()


class _enable_get_wd_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_wd_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_wd_called_within_step = False


# class CosineWDSchedule(WDScheduler):

#     def __init__(
#         self,
#         optimizer: Optimizer,
#         ref_wd: float,
#         T_max: int,
#         final_wd: float = 0.0
#     ) -> None:
#         assert ref_wd >= 0
#         assert final_wd >= 0
#         assert T_max >= 0

#         self.ref_wd = ref_wd
#         self.T_max = T_max
#         self.final_wd = final_wd
#         super(CosineWDSchedule, self).__init__(optimizer=optimizer)

#     def get_wd(self):
#         progress = self._step_count / self.T_max
#         new_wd = self.final_wd + \
#             (self.ref_wd - self.final_wd) * 0.5 * \
#             (1. + math.cos(math.pi * progress))

#         if self.final_wd <= self.ref_wd:
#             new_wd = max(self.final_wd, new_wd)
#         else:
#             new_wd = min(self.final_wd, new_wd)

#         return [new_wd for _ in self.optimizer.param_groups]
