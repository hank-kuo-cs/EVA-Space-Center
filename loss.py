import numpy as np
import torch
from config import BATCH_SIZE, LABEL_TYPE, LABEL_NUM, GAMMA_RANGE, GAMMA_RADIUS, GAMMA_UNIT


def get_gamma(output):
    dist = 0
    for i in range(LABEL_NUM):
        dist += ((output[i] * (GAMMA_RANGE + GAMMA_RADIUS)) ** 2).item()

    dist = np.array(dist)
    dist = np.sqrt(dist)

    return dist


def get_error_percentage(output, target):
    output = output.double()
    error_percentage = []

    for i in range(LABEL_NUM):
        error = (abs(output[i] - target[i])).item()
        error_percentage.append(error)

    gamma_error = abs(output[0] - target[0]).item()

    return np.array(error_percentage), gamma_error


class BCMSELoss(torch.nn.Module):
    def __init__(self):
        super(BCMSELoss, self).__init__()

    def forward(self, outputs, targets):
        constant_penalties = []

        for i in range(BATCH_SIZE):
            constant_penalty = np.array([0, 0], dtype=np.float64)

            for j in range(1, 3):
                constant_penalty[j - 1] = abs(outputs[i][j] // 1)
                outputs[i][j] = torch.remainder(outputs[i][j], 1)

                if abs(outputs[i][j] - targets[i][j]) > 0.5:
                    targets[i][j] = 1 + targets[i][j] if targets[i][j] < outputs[i][j] else -1 + targets[i][j]

            constant_penalties.append(constant_penalty)

        constant_penalties = np.array(constant_penalties).sum() / BATCH_SIZE
        amount_loss = torch.tensor(constant_penalties, dtype=torch.double)

        mse_loss = torch.nn.MSELoss()(outputs, targets)
        loss = torch.add(mse_loss, amount_loss)

        return loss


class MoonLoss(torch.nn.Module):
    def __init__(self):
        super(MoonLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = torch.nn.MSELoss()(outputs, targets)
        return loss
