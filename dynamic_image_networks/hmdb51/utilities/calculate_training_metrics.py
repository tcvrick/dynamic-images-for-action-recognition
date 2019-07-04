import torch
import numpy as np


def calculate_accuracy(y_pred, y_true):
    # Inspired from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
    _, predicted = torch.max(y_pred, 1)
    acc = (predicted == y_true).sum().item() / len(y_pred)
    return acc
