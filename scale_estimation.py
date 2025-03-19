import numpy as np
import torch
import util_gau
from util_gau import GaussianData
import numpy as np
from nopo_transforms import *
import json
import os
import matplotlib.pyplot as plt
import scipy.spatial.transform

# T_AB is transform from frame B to frame A

# Load transformation matrices from JSON file
DATE = "25_03_19"
PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/"
tf_dict = {}
poses_path = PATH + 'poses/'

for file_name in os.listdir(poses_path):
    if file_name.endswith('.json'):
        # print(file_name)
        index0 = file_name.split('_')[1]
        index1 = file_name.split('_')[2].split('.')[0]
        with open(os.path.join(poses_path, file_name), 'r') as f:
            tf_dict[index0 + "_" + index1] = np.array(json.load(f)).squeeze()

# tf_25_0 = np.array([[ 9.2080e-01,  4.1117e-02, -3.8786e-01, -2.7666e-01],
#           [-4.8231e-02,  9.9880e-01, -8.6195e-03,  2.3985e-01],
#           [ 3.8704e-01,  2.6644e-02,  9.2168e-01,  3.1883e-03],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

# tf_0_50 = np.array([[ 7.8877e-01, -1.0122e-01,  6.0630e-01,  8.1825e-01],
#           [ 1.1857e-01,  9.9288e-01,  1.1505e-02, -2.2407e-01],
#           [-6.0315e-01,  6.2814e-02,  7.9515e-01,  1.5644e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

# print("a", np.linalg.norm(tf_25_0[:3, 3]))
# print(np.linalg.norm(tf_0_50[:3, 3]))

# tf_0_25 = tf_dict["0_25"]
# tf_0_25 = np.array(tf_0_25).squeeze()


# # print(np.linalg.inv(tf_25_0)- tf_0_25)
# r_0_25 = tf_0_25[:3, :3]
# r_25_0 = tf_25_0[:3, :3]

# t_0_25 = tf_0_25[:3, 3]
# t_25_0 = tf_25_0[:3, 3]

# print(r_25_0.T @ t_25_0, "*s = ", -t_0_25)
# # Solve least squares for scalar value s
# s_25 = np.linalg.lstsq((r_25_0.T @ t_25_0).reshape(-1, 1), -t_0_25.reshape(-1, 1), rcond=None)[0].item()
# print(r_25_0.T @ t_25_0*s_25, " ?= ", -t_0_25)

# print("s_25", s_25)

# # print(tf_0_25)
# # print(np.linalg.inv(tf_25_0))

# print(t_0_25)
# print(t_25_0.T)

# tf_25_50_star = np.linalg.inv(tf_0_25) @ tf_0_50

# tf_25_50 = np.array(tf_dict["25_50"]).squeeze()
# print(tf_25_50_star)
# print(tf_25_50)

# s_25_star = np.linalg.lstsq((tf_25_50_star[:3,3]).reshape(-1, 1), -tf_25_50[:3,3].reshape(-1, 1), rcond=None)[0].item()

# print(tf_25_50_star[:3,3])
# print(tf_25_50[:3,3])
# print(1/s_25_star)

def estimate_scale(tf_01, tf_10, tf_02, tf_12) -> float:
    r_01 = tf_01[:3, :3]
    r_10 = tf_10[:3, :3]
    r_02 = tf_02[:3, :3]

    t_01 = tf_01[:3, 3]
    t_10 = tf_10[:3, 3]
    t_02 = tf_02[:3, 3]

    # tf_12 = np.linalg.inv(tf_01) @ tf_02
    # r_12 = tf_12[:3, :3]
    # t_12 = tf_12[:3, 3]
    # print(r_01, r_10.T)
    # print(t_01, -r_10.T @ t_10)

    s = np.linalg.lstsq((r_10.T @ t_10).reshape(-1, 1), -t_01.reshape(-1, 1), rcond=None)[0].item()
    # s = np.linalg.norm(t_01) / np.linalg.norm(r_10.T @ t_10)
    error = np.linalg.norm((r_10.T @ t_10 * s) + t_01)
    print(f"Least squares error: {error}")

    return s

# tf_10_0 = np.array([[ 9.9256e-01,  1.3876e-02, -1.2093e-01, -6.6544e-01],
#           [-1.4295e-02,  9.9989e-01, -2.5974e-03,  2.9687e-01],
#           [ 1.2088e-01,  4.3067e-03,  9.9266e-01, -1.4825e-02],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
# s_10 = estimate_scale(tf_dict["0_10"], tf_10_0)
# print(s_10)


