import numpy as np
import torch


def rodrigues(rot_x, rot_y, rot_z, inv_flag):
    # rotation_vector = torch.tensor([rot_x, rot_y, rot_z],
    #                                dtype=torch.float32)
    # theta = torch.norm(rotation_vector)
    # if theta.item() == 0:
    #     return torch.eye(3, dtype=torch.float32)
    # #
    # rotation_axis = rotation_vector / theta
    # R = torch.tensor([[0., -rotation_axis[2], rotation_axis[1]],
    #                   [rotation_axis[2], 0, -rotation_axis[0]],
    #                   [-rotation_axis[0], rotation_axis[1], 0.]], dtype=torch.float32)
    # #
    # I = torch.eye(3, dtype=torch.float32)
    # rotation_matrix = I + torch.sin(theta) * R + (1 - torch.cos(theta)) * R @ R

    if not inv_flag:
        sin_rx, cos_rx = torch.sin(rot_x), torch.cos(rot_x)
        sin_ry, cos_ry = torch.sin(rot_y), torch.cos(rot_y)
        sin_rz, cos_rz = torch.sin(rot_z), torch.cos(rot_z)
        R_x = torch.tensor([[1, 0, 0],
                            [0, cos_rx, -sin_rx],
                            [0, sin_rx, cos_rx]])
        R_y = torch.tensor([[cos_ry, 0, sin_ry],
                            [0, 1, 0],
                            [-sin_ry, 0, cos_ry]])
        R_z = torch.tensor([[cos_rz, -sin_rz, 0],
                            [sin_rz, cos_rz, 0],
                            [0, 0, 1]])
        rotation_matrix = R_x @ R_y @ R_z
    else:
        # rotation_matrix = torch.linalg.pinv(rotation_matrix)
        sin_rx, cos_rx = np.sin(rot_x), np.cos(rot_x)
        sin_ry, cos_ry = np.sin(rot_y), np.cos(rot_y)
        sin_rz, cos_rz = np.sin(rot_z), np.cos(rot_z)

        R_x = torch.tensor([[1, 0, 0],
                            [0, cos_rx, -sin_rx],
                            [0, sin_rx, cos_rx]])
        R_y = torch.tensor([[cos_ry, 0, sin_ry],
                            [0, 1, 0],
                            [-sin_ry, 0, cos_ry]])
        R_z = torch.tensor([[cos_rz, -sin_rz, 0],
                            [sin_rz, cos_rz, 0],
                            [0, 0, 1]])
        rotation_matrix = R_x @ R_y @ R_z
        rotation_matrix = rotation_matrix.T
    return rotation_matrix


fx = 1.
fy = 1.
cx = 1.
cy = 1.
x = 0.33533
y = 0.4494
z = 0.2525
rx = torch.tensor([30.])
ry = torch.tensor([30.])
rz = torch.tensor([30.])
tx = torch.tensor([1.])
ty = torch.tensor([0.])
tz = torch.tensor([0.])
dim_equalizer = torch.tensor([0., 0., 0., 1.]).unsqueeze(0)
I = torch.eye(3, dtype=torch.float32)
I_full = torch.cat([I, torch.zeros((3, 1))], dim=1)

# to 2d
R = rodrigues(rx, ry, rz, 0)  # Rotation matrix
t = torch.tensor([[tx], [ty], [tz]], dtype=torch.float32)  # Translation  matrix
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera Matrix

RT = torch.cat([R, t], dim=1)
RT = torch.cat([RT, dim_equalizer], dim=0)
points_3d_orginal = torch.tensor([[x, y, z, 1.]])
points_3d_transformed = (K @ I_full @ RT @ points_3d_orginal.T)

# to 3d
K_inv = torch.linalg.pinv(K)
points_3d_transformedw = torch.cat([points_3d_transformed, torch.ones((1, 1))], dim=0)
points_3d_transformed_world = K_inv @ I_full @ points_3d_transformedw
points_3d_transformed_world = torch.cat([points_3d_transformed_world, torch.ones((1, 1))], dim=0)
RT_inv = torch.cat([R.T, -R.T @ t], dim=1)
RT_inv = torch.cat([RT_inv, dim_equalizer], dim=0)
points_3d_orginal = RT_inv @ points_3d_transformed_world
print(points_3d_orginal)
