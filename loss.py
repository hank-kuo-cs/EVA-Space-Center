import numpy as np
from config import *


def get_error_percentage(output, target):
    output = output.double()

    error_percentage = [0, 0, 0]

    error_percentage[0] = (abs(output[0] - target[0]) / target[0]).item()

    output[2] /= 2
    target[2] /= 2

    for i in range(1, 3):
        if output[i] < 0:
            output[i] = output[i] % 1
        dis = abs(output[i] % 1 - target[i])
        dis = dis if dis < 0.5 else 1 - dis
        error_percentage[i] = (dis / target[i]).item()

    return np.array(error_percentage)


class MoonMSELoss(torch.nn.Module):
    def __init__(self):
        super(MoonMSELoss, self).__init__()

    def forward(self, outputs, targets):
        batch_circle_num = []

        for i in range(BATCH_SIZE):
            circle_num = np.array([0, 0], dtype=np.float64)

            targets[i][1] /= 2 * np.pi
            targets[i][2] /= np.pi

            outputs[i][1] = torch.div(outputs[i][1], 2 * np.pi)
            outputs[i][2]= torch.div(outputs[i][2], np.pi)

            for j in range(1, 3):
                if outputs[i][j] < 0:
                    circle_num[j - 1] = outputs[i][j] // -1
                    outputs[i][j] = torch.remainder(outputs[i][j], 1)

                dis = abs(outputs[i][j] % 1 - targets[i][j])
                if dis > 0.5:
                    if outputs[i][j] % 1 < targets[i][j]:
                        targets[i][j] = outputs[i][j] + 1 - dis
                    else:
                        targets[i][j] = outputs[i][j] - 1 + dis
                circle_num[j - 1] += outputs[i][j] // 1
                outputs[i][j] = torch.remainder(outputs[i][j], 1)

            outputs[i][2] = torch.mul(outputs[i][2], 2)
            circle_num[1] *= 2

            batch_circle_num.append(circle_num)

        batch_circle_num = np.array(batch_circle_num).sum() / BATCH_SIZE / 2 / 100
        amount_loss = torch.tensor(batch_circle_num, dtype=torch.double)

        mse_loss = torch.nn.MSELoss()(outputs, targets)
        loss = torch.add(mse_loss, amount_loss)

        return loss


a = torch.tensor(5.6)
torch.remainder(a, 1)
print(a)