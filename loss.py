import numpy as np
from config import *

from torch.utils.data import DataLoader
from data import MoonDataset


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


def dynamic_constant_penalty(outputs, targets, constant_loss):
    direction_loss = [[], [], []]
    criterion = torch.nn.MSELoss()
    for i in range(BATCH_SIZE):
        for j in range(0, 4, 3):
            direction_loss[0].append(np.array(criterion(outputs[i][j], targets[i][j]).cpu().detach().numpy()))
            for k in range(1, 3):
                direction_loss[1].append(
                    np.array(criterion(outputs[i][j + k], targets[i][j + k]).cpu().detach().numpy()))
        for j in range(6, 9):
            direction_loss[2].append(np.array(criterion(outputs[i][j], targets[i][j]).cpu().detach().numpy()))

    dir_percentage = []
    for i in range(3):
        dir_percentage.append(np.array(direction_loss[i]).sum())

    if (dir_percentage[0] > dir_percentage[1]) and (dir_percentage[0] > dir_percentage[2]):
        constant_loss = 0
    elif (dir_percentage[0] > dir_percentage[1]) and (dir_percentage[0] < dir_percentage[2]):
        constant_loss = constant_loss[1]
    elif (dir_percentage[0] < dir_percentage[1]) and (dir_percentage[0] > dir_percentage[2]):
        constant_loss = constant_loss[0]
    else:
        constant_loss = constant_loss[0] + constant_loss[1]

    return constant_loss


def set_constant_oenalty_mode(mode, outputs, targets, mse_loss, constant_loss):
    if mode == 'dcp':
        amount_loss = dynamic_constant_penalty(outputs, targets, constant_loss)
    elif mode == 'cp':
        amount_loss = constant_loss.sum()
    elif mode == 'little_cp':
        amount_loss = constant_loss.sum() * mse_loss.clone().double().detach().cpu()
    elif mode == 'no_cp':
        amount_loss = 0

    return amount_loss


class BCMSELoss(torch.nn.Module):
    def __init__(self):
        super(BCMSELoss, self).__init__()

    def forward(self, outputs, targets):
        ectra_angles = []
        extra_scalars = []

        for i in range(BATCH_SIZE):

            for j in range(0, 4, 3):
                extra_angle = np.array([0, 0], dtype=np.float64)  # angle constant

                for k in range(1, 3):
                    # print((outputs[i] == outputs[i][j + k]).nonzero())
                    extra_angle[k - 1] = abs(outputs[i][j + k] // 1)
                    outputs[i][j + k] = torch.remainder(outputs[i][j + k], 1)

                    if abs(outputs[i][j + k] - targets[i][j + k]) > 0.5:
                        if targets[i][j + k] < outputs[i][j + k]:
                            targets[i][j + k] = 1 + targets[i][j + k]
                        else:
                            targets[i][j + k] = -1 + targets[i][j + k]

                ectra_angles.append(extra_angle)

            outputs[i][6: 9], extra_scalar = get_scalar(outputs[i][6: 9])
            extra_scalars.append(extra_scalar)

        constant_penalty = [ectra_angles, extra_scalars]
        constant_penalties = np.array([.0, .0])
        constant_loss = np.array([.0, .0])
        for i in range(2):
            constant = np.array(constant_penalty[i]).sum()
            constant_penalties[i] = constant / BATCH_SIZE / CONSTANT_WEIGHT
            constant_loss[i] = constant_penalties[i]

        mse_loss = torch.nn.MSELoss()(outputs, targets)
        amount_loss = set_constant_oenalty_mode(CONSTANT_PENALITY_MODE, outputs, targets, mse_loss, constant_loss)
        loss = torch.add(mse_loss, amount_loss)

        return loss


def sphere2cartesian(ball_coordinate_vector):

    gamma = ball_coordinate_vector[:, 0]
    theta = ball_coordinate_vector[:, 1]
    phi = ball_coordinate_vector[:, 2]
    x = gamma * torch.sin(theta) * torch.cos(phi)
    y = gamma * torch.sin(theta) * torch.sin(phi)
    z = gamma * torch.cos(theta)
    cassette_coordinate_vector = torch.tensor([x, y, z], dtype=torch.double, device=DEVICE, requires_grad=True)

    return cassette_coordinate_vector


def get_scalar(vector_list):
    scalar = torch.sqrt(torch.dot(vector_list, vector_list))
    normal_vector = torch.remainder(vector_list, scalar.clone().detach())

    return normal_vector, scalar


def unnormalize(normalized_vector):
    remainder = torch.tensor([MOON_RADIUS, 0, 0, 0, 0, 0, -1, -1, -1],
                             dtype=torch.double, device=DEVICE, requires_grad=False)
    limit = torch.tensor(LIMIT, dtype=torch.double, device=DEVICE, requires_grad=False)
    remind_vector = torch.add(normalized_vector, remainder)
    unnormalize_vector = torch.mul(remind_vector, limit)

    return unnormalize_vector


class CosSimiSphericalLoss(torch.nn.Module):
    def __init__(self):
        super(CosSimiSphericalLoss, self).__init__()

    def forward(self, outputs, targets):
        constant_penalties = []
        similarity = torch.tensor([], dtype=torch.double, device=DEVICE, requires_grad=True)

        unnormalize_outputs = unnormalize(outputs)
        unnormalize_targets = unnormalize(targets)
        camera_outputs, optic_outputs, nor_outputs = torch.split(unnormalize_outputs, 3, dim=1)
        camera_targets, optic_targets, nor_targets = torch.split(unnormalize_targets, 3, dim=1)
        camera_cas_outputs = sphere2cartesian(camera_outputs)
        optic_cas_outputs = sphere2cartesian(optic_outputs)
        camera_cas_targets = sphere2cartesian(camera_targets)
        optic_cas_targets = sphere2cartesian(optic_targets)

        cas_outputs = [camera_cas_outputs, optic_cas_outputs, nor_outputs]
        cas_targets = [camera_cas_targets, optic_cas_targets, nor_targets]
        for j in range(3):
            unit_cas_outputs, outputs_scalar = get_scalar(cas_outputs[j])
            unit_cas_targets, targets_scalar = get_scalar(cas_targets[j])
            scalar_tmp = (targets_scalar - outputs_scalar).cuda().clone().detach().requires_grad_(True)
            vector_tmp = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(
                torch.reshape(unit_cas_targets, (1, 3)),
                torch.reshape(unit_cas_outputs, (1, 3))).cuda().clone().detach().requires_grad_(True)
            if j == 0:
                constant_penalties = torch.tensor(scalar_tmp,
                                                  dtype=torch.double, device=DEVICE, requires_grad=True)
                similarity = torch.tensor(vector_tmp, dtype=torch.double, device=DEVICE, requires_grad=True)
            else:
                constant_penalties = torch.add(constant_penalties, scalar_tmp)
                similarity = torch.add(similarity, vector_tmp)

        print("similarity_loss: {}".format(torch.remainder(similarity, BATCH_SIZE).item()))

        constant_loss = torch.remainder(constant_penalties, BATCH_SIZE)
        similarity_loss = torch.remainder(similarity, BATCH_SIZE)
        loss = torch.add(similarity_loss, constant_loss)

        return loss


if __name__ == '__main__':
    train_set = MoonDataset('train')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for i, data in enumerate(train_loader):
        _, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        unnormalize_targets = unnormalize(labels)
        print("unnormalize: {}".format(unnormalize_targets))
        camera_targets, optic_targets, nor_targets = torch.split(unnormalize_targets, 3, dim=1)
        camera_cas_targets = sphere2cartesian(camera_targets)
        print("ball: {}".format(camera_targets))
        print("cas: {}".format(camera_cas_targets))
        exit(1)
