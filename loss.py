import numpy as np
from config import *


def get_error_percentage(output, target):
    output = output.double()

    error_percentage = [0, 0, 0]

    error_percentage[0] = (abs(output[0] - target[0]) / GAMMA_RANGE).item()

    for i in range(1, 3):
        if output[i] < 0:
            output[i] = output[i] % (2 * np.pi)

        dis = abs(output[i] - target[i])
        error_percentage[i] = ((2 * np.pi - dis) / (2 * np.pi)).item() if dis > np.pi else (dis / (2 * np.pi)).item()

    return np.array(error_percentage)


class MoonMSELoss(torch.nn.Module):
    def __init__(self):
        super(MoonMSELoss, self).__init__()

    def forward(self, outputs, targets):
        constant_penalties = []

        for i in range(BATCH_SIZE):
            constant_penalty = np.array([0, 0], dtype=np.float64)

            for j in range(1, 3):
                constant_penalty[j - 1] = abs(outputs[i][j] // (2 * np.pi))
                outputs[i][j] = torch.remainder(outputs[i][j], 2 * np.pi)

                if abs(outputs[i][j] - targets[i][j]) > np.pi:
                    targets[i][j] = 2 * np.pi + targets[i][j] if targets[i][j] < outputs[i][j] else -2 * np.pi + targets[i][j]

            constant_penalties.append(constant_penalty)

        constant_penalties = np.array(constant_penalties).sum() / BATCH_SIZE / 2 / 10000
        amount_loss = torch.tensor(constant_penalties, dtype=torch.double)

        mse_loss = torch.nn.MSELoss()(outputs, targets)
        loss = torch.add(mse_loss, amount_loss)

        return loss
