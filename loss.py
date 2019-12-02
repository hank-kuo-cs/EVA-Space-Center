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


def ball_coordinates_to_cassette_coordinates(ball_coordinate_vector):
    cassette_coordinate_vector = torch.tensor([.0, .0, .0], dtype=torch.double, requires_grad=True)

    gamma = ball_coordinate_vector[0]
    theta = ball_coordinate_vector[1]
    phi = ball_coordinate_vector[2]
    cassette_coordinate_vector[0] = gamma * torch.sin(theta) * torch.cos(phi)
    cassette_coordinate_vector[1] = gamma * torch.sin(theta) * torch.sin(phi)
    cassette_coordinate_vector[2] = gamma * torch.cos(theta)

    return cassette_coordinate_vector


def get_scalar(vector_list):
    scalar = torch.sqrt(torch.dot(vector_list, vector_list))
    normal_vector = torch.remainder(vector_list, scalar.clone().detach())

    return normal_vector, scalar


class CosSimiBCLoss(torch.nn.Module):
    def __init__(self):
        super(CosSimiBCLoss, self).__init__()

    def forward(self, outputs, targets):
        constant_penalties = []
        similarity = []

        for i in range(BATCH_SIZE):
            camera_cas_outputs = ball_coordinates_to_cassette_coordinates(outputs[i][0:3])
            optic_cas_outputs = ball_coordinates_to_cassette_coordinates(outputs[i][3:6])
            nor_cas_outputs = outputs[i][6:9]
            camera_cas_targets = ball_coordinates_to_cassette_coordinates(targets[i][0:3])
            optic_cas_targets = ball_coordinates_to_cassette_coordinates(targets[i][3:6])
            nor_cas_targets = targets[i][6:9]

            cas_outputs = [camera_cas_outputs, optic_cas_outputs, nor_cas_outputs]
            cas_targets = [camera_cas_targets, optic_cas_targets, nor_cas_targets]
            for j in range(3):
                unit_cas_outputs, outputs_scalar = get_scalar(cas_outputs[j])
                unit_cas_targets, targets_scalar = get_scalar(cas_targets[j])
                constant_penalties.append((targets_scalar - outputs_scalar))
                similarity.append(torch.nn.CosineSimilarity(dim=1, eps=1e-6)(
                                        torch.reshape(unit_cas_targets, (1, 3)),
                                        torch.reshape(unit_cas_outputs, (1, 3))))

        print("similarity_loss: {}".format(torch.mean(torch.stack(torch.tensor(similarity, device=DEVICE))).item()))

        constant_loss = torch.mean(torch.stack(torch.tensor(constant_penalties, device=DEVICE)))
        similarity_loss = torch.mean(torch.stack(torch.tensor(similarity, device=DEVICE)))
        loss = torch.add(similarity_loss, constant_loss)

        return loss
