import numpy as np
import torch
from config import BATCH_SIZE, LABEL_TYPE, LABEL_NUM


def get_error_percentage(output, target):
    output = output.double()

    error_percentage = (abs(output - target[:3])).item()

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

    return np.array(error_percentage)


def get_gamma_error_pencentage(output, c_gamma):
    output = output.double()

    dist = 0
    for i in range(LABEL_NUM):
        dist += output[i] ** 2

    dist = np.array(dist)
    dist = np.sqrt(dist)

    error = abs(dist - c_gamma)

    return error


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
        # for i in range(BATCH_SIZE):
        #     targets[i] = targets[i][:3]

        loss = torch.nn.MSELoss()(outputs, targets[:][:3])
        return loss
