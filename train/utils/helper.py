import datetime

import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from utils import lang_enum


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1 - probs) * (1 - labels)
    alpha = torch.ones_like(logits) * alpha
    alpha = torch.where(labels == 1, alpha, 1 - alpha)
    loss = -alpha * torch.pow(1 - pt, gamma) * torch.log(pt + 1e-9)
    return loss


class InverseSquareRootLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr,
        target_lr,
        steps,
        linear_begin=0,
        linear_steps=0,
        warmup_steps=1500,
        x_value=4.0,
        last_epoch=-1,
        rsqrt_initial_value=0.5,
    ):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.x_value = x_value
        self.x_value_steps = steps
        self.x_lr_value = target_lr
        self.rsqrt_initial_value = rsqrt_initial_value

        self.first_rsqrt = rsqrt_initial_value**-0.5
        self.last_rsqrt = x_value**-0.5
        self.dist_rsqrt = self.first_rsqrt - self.last_rsqrt

        self.linear_begin = linear_begin
        self.linear_steps = linear_steps
        self.linear_initial_lr = self.calc_rsqrt(self.linear_begin)

        super().__init__(optimizer, last_epoch)

    def calc_rsqrt(self, step):
        step_rsqrt = (
            self.rsqrt_initial_value + (step / self.x_value_steps) * (self.x_value - self.rsqrt_initial_value)
        ) ** -0.5
        step_rsqrt = step_rsqrt - self.last_rsqrt
        step_rsqrt = step_rsqrt / self.dist_rsqrt
        step_rsqrt = step_rsqrt * self.initial_lr
        lr = step_rsqrt + self.x_lr_value
        return lr

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            lr = self.initial_lr * self._step_count / self.warmup_steps
        elif self._step_count > self.warmup_steps + self.linear_begin:
            if self._step_count > self.warmup_steps + self.linear_begin + self.linear_steps:
                lr = 0.0
            else:
                step = self._step_count - self.warmup_steps - self.linear_begin
                lr = self.linear_initial_lr - (step / self.linear_steps) * self.linear_initial_lr
        elif self._step_count <= self.x_value_steps + self.warmup_steps:
            step = self._step_count - self.warmup_steps
            lr = self.calc_rsqrt(step)
        else:
            lr = self.x_lr_value

        return [lr]


def augment(text, lines_num_range=(5, 120)):
    text_list = text.split("\n")
    text_list = [line for line in text_list if line.strip()]
    random_len = min(np.random.randint(*lines_num_range), len(text_list))
    random_start = np.random.randint(0, len(text_list) - random_len) if len(text_list) > random_len else 0
    return "\n".join(text_list[random_start : random_start + random_len])


def generate_experiment_name(hyperparameters, prefix=None):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hyperparameters_str = "_".join(f"{key}={value}" for key, value in hyperparameters.items())
    experiment_name = f"{date_time}_{hyperparameters_str}"

    if prefix is not None:
        experiment_name = f"{prefix}_{experiment_name}"

    return experiment_name


def compute_class_weights(df, alpha=6.0):
    groups = df.groupby("language_tag").count()
    labels = groups.index.values
    counts = groups.language.values

    counts = (counts.max() - counts) + counts.min()
    inv_freq = 1.0 / counts
    inv_freq_norm = inv_freq / np.sum(inv_freq)
    exponential_weights = np.power(counts, alpha) / np.sum(np.power(counts, alpha))
    combined_weights = inv_freq_norm * exponential_weights
    normalized_weights = combined_weights / np.sum(combined_weights)
    normalized_weights = normalized_weights * 1000 + 1

    weights = np.ones_like(labels, dtype=np.float32)
    for label, weight in zip(labels, normalized_weights):
        weights[lang_enum.languages2num[label]] = weight
    return weights / weights.mean()


def compute_class_weights_sqrt(df, gamma=2.0):
    groups = df.groupby("language_tag").count()
    labels = groups.index.values
    counts = groups.language.values

    sqrt_inverse_freq = np.sqrt(1 / counts)
    normalized_weights = sqrt_inverse_freq / sqrt_inverse_freq.sum()
    normalized_weights = normalized_weights**gamma
    normalized_weights /= normalized_weights.mean()

    weights = np.ones_like(labels, dtype=np.float32)
    for label, weight in zip(labels, normalized_weights):
        weights[lang_enum.languages2num[label]] = weight
    return weights / weights.mean()


def compute_linear_class_weights(df):
    groups = df.groupby("language_tag").count()
    labels = groups.index.values
    counts = groups.language.values

    counts = (counts.max() - counts) + counts.min()
    weights = counts / np.sum(counts)

    weights = np.ones_like(labels, dtype=np.float32)
    for label, weight in zip(labels, weights):
        weights[lang_enum.languages2num[label]] = weight
    return weights / weights.mean()
