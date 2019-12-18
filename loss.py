import numpy as np
import torch
from config import BATCH_SIZE, LABEL_TYPE, LABEL_NUM, GAMMA_RANGE, GAMMA_RADIUS


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

    c_gamma = target[3].item()

    dist = get_gamma(output)

    gamma_error = abs(dist - c_gamma)
    gamma_error = (gamma_error - GAMMA_RADIUS) / GAMMA_RANGE

    # for i in range(1, 3):
    #     output[i] = output[i] % 1
    #
    #     dis = abs(output[i] - target[i])
    #     error_percentage[i] = (1 - dis).item() if dis > 0.5 else dis.item()
    #
    # for i in range(3, 5):
    #     output[i] = output[i] % 1
    #
    #     dis = abs(output[i] - target[i])
    #     error_percentage[i] = (1 - dis).item() if dis > 0.5 else dis.item()

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
        loss = torch.nn.MSELoss()(outputs, targets[:, :3])
        return loss
