import numpy as np
from config import *


def get_error_percentage(output, target):
    output = output.double()

    error_percentage = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    output[6:9], _ = get_scalar(output[6:9])

    for i in range(0, 4, 3):
        error_percentage[i] = (abs(output[i] - target[i])).item()
        for j in range(1, 3):
            if output[i + j] < 0:
                output[i + j] = output[i + j] % 1

            dis = abs(output[i + j] - target[i + j])
            error_percentage[i + j] = (1 - dis).item() if dis > 0.5 else dis.item()

    for i in range(6, 9):
        error_percentage[i] = (abs(output[i] - target[i])).item()

    return np.array(error_percentage)


def get_scalar(vector_list):
    tmp = .0
    for i in range(3):
        tmp += vector_list[i] ** 2
    scalar = np.sqrt(tmp)
    normal_vector = torch.remainder(vector_list, torch.tensor(scalar, dtype=np.float64))

    return normal_vector, scalar


class BCMSELoss(torch.nn.Module):
    def __init__(self):
        super(BCMSELoss, self).__init__()

    def forward(self, outputs, targets):
        constant_penalties = []

        for i in range(BATCH_SIZE):

            for j in range(0, 4, 3):
                constant_penalty = np.array([0, 0], dtype=np.float64)

                for k in range(1, 3):
                    # print((outputs[i] == outputs[i][j + k]).nonzero())
                    constant_penalty[k - 1] = abs(outputs[i][j + k] // 1)
                    outputs[i][j + k] = torch.remainder(outputs[i][j + k], 1)

                    if abs(outputs[i][j + k] - targets[i][j + k]) > 0.5:
                        if targets[i][j + k] < outputs[i][j + k]:
                            targets[i][j + k] = 1 + targets[i][j + k]
                        else:
                            targets[i][j + k] = -1 + targets[i][j + k]

                constant_penalties.append(constant_penalty)

            outputs[i][6: 9], scalar = get_scalar(outputs[i][6: 9])

        constant_penalties = (np.array(constant_penalties).sum() + scalar) / BATCH_SIZE / CONSTANT_WEIGHT
        amount_loss = torch.tensor(constant_penalties, dtype=torch.double)

        mse_loss = torch.nn.MSELoss()(outputs, targets)
        loss = torch.add(mse_loss, amount_loss)

        return loss


class BCL1Loss(torch.nn.Module):
    def __init__(self):
        super(BCL1Loss, self).__init__()

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

        constant_penalties = np.array(constant_penalties).sum() / BATCH_SIZE / 2 / 10000
        amount_loss = torch.tensor(constant_penalties, dtype=torch.double)

        l1_loss = torch.nn.L1Loss()(outputs, targets)
        loss = torch.add(l1_loss, amount_loss)

        return loss
