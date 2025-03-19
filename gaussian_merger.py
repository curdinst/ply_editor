# import torch
# import glfw
# import OpenGL.GL as gl
# from imgui.integrations.glfw import GlfwRenderer
# import imgui
# import numpy as np
# import util
# import imageio
import torch
import util_gau
from util_gau import GaussianData
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import scipy.spatial.transform
from nopo_transforms import *
from scale_estimation import *

COLORED = True
SCALED = True
GTSCALED = False
NOPOSCALED = False
assert (SCALED and GTSCALED) == False

# DATE = "25_03_13_scale_inv_false"
DATE = "25_03_19"

FRAMES = [0, 10, 20]
COLOR_BOOST = 0.4
IMGPATH = "/home/curdin/master_thesis/images/"
DATASET_PATH = "/home/curdin/noposplat/NoPoSplat/datasets/re10k/test/"
DATASET_FILES = ["000000.torch",  "000001.torch",  "000002.torch"]
# KEY = "1214f2a11a9fc1ed"
KEY = "c48f19e2ffa52523"

PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/"

noposcaled = "_unscaled_" if NOPOSCALED else "_"

OUTPUT_PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/gaussians/"

file0 = KEY + noposcaled + str(FRAMES[0]) + "_" + str(FRAMES[1]) + ".ply"
file1 = KEY + noposcaled + str(FRAMES[1]) + "_" + str(FRAMES[2]) + ".ply"
# file2 = KEY + noposcaled + str(FRAMES[2]) + "_" + str(FRAMES[3]) + ".ply"

print("Files: ", file0, "\n", file1)

# Load transformation matrices from JSON file
tf_dict = {}
poses_path = PATH + 'poses/'

for file_name in os.listdir(poses_path):
    if file_name.endswith('.json'):
        # print(file_name)
        index0 = file_name.split('_')[1]
        index1 = file_name.split('_')[2].split('.')[0]
        with open(os.path.join(poses_path, file_name), 'r') as f:
            tf_dict[index0 + "_" + index1] = json.load(f)

# Convert the transformation matrices from lists to numpy arrays
for key in tf_dict:
    tf_dict[key] = np.array(tf_dict[key]).squeeze()
    # print(np.array(tf_dict[key]))
    # print(tf_dict[key][:3, 3])
    # break

# file0 = "1214f2a11a9fc1ed_10-60.ply"
# file0 = "1214f2a11a9fc1ed_60_120.ply"
# file1 = "1214f2a11a9fc1ed_120_180.ply"
# file3 = "1214f2a11a9fc1ed_180_240.ply"

# file0 = "1214f2a11a9fc1ed_90_120.ply"
# file1 = "1214f2a11a9fc1ed_120_150.ply"
# file0 = "1214f2a11a9fc1ed_120_150.ply"
# file1 = "1214f2a11a9fc1ed_150_180.ply"
# file2 = "1214f2a11a9fc1ed_180_210.ply"

ply_frame_0 = file0.split('_')[1]
print(file0)
print(file0.split('_'))
ply_frame_1 = file1.split('_')[1]
ply_frame_2 = file1.split('_')[2].split('.')[0]
print("Last two digits of file0:",ply_frame_0, ply_frame_1)
files = [file0, file1]

file_paths = [OUTPUT_PATH + file for file in files]
# path_file0 = OUTPUT_PATH + file0
# path_file1 = OUTPUT_PATH + file1


for file in DATASET_FILES:
    dataset_filepath = DATASET_PATH + file
    dataset = torch.load(dataset_filepath)
    keys = [dataset[i]["key"] for i in range(len(dataset))]
    if KEY in keys:
        print(f"Found key {KEY} in {file} at idx {keys.index(KEY)}")
        idx = keys.index(KEY)
        data = dataset[idx]
        break

print(data.keys())
transfomrs = data["cameras"]
print("num frames: ", len(transfomrs))

frame0 = transfomrs[FRAMES[0]][6:].reshape(3,4)
frame1 = transfomrs[FRAMES[1]][6:].reshape(3,4)
frame2 = transfomrs[FRAMES[2]][6:].reshape(3,4)
# frame3 = transfomrs[FRAMES[3]][6:].reshape(3,4)
frame0 = torch.vstack([frame0, torch.tensor([0, 0, 0, 1])])
frame1 = torch.vstack([frame1, torch.tensor([0, 0, 0, 1])])
frame2 = torch.vstack([frame2, torch.tensor([0, 0, 0, 1])])
# frame3 = torch.vstack([frame3, torch.tensor([0, 0, 0, 1])])

# print(frame0)
# print(frame1)
tf_10 = np.linalg.inv(frame1) @ frame0
tf_21 = np.linalg.inv(frame2) @ frame1
# tf_32 = np.linalg.inv(frame3) @ frame2

t1 = np.linalg.norm(tf_10[:3, 3])
t2 = np.linalg.norm(tf_21[:3, 3])
# t3 = np.linalg.norm(tf_32[:3, 3])
print("norm 1", t1)
print("norm 2", t2)
print(t2/t1)

img1, img2, img3 = 10, 60, 120
img1, img2, img3 = 60, 120, 180
img1, img2, img3 = 90, 120, 150
img4, img5 = 280, 240


image0 = data["images"][img1]
image1 = data["images"][img2]
image2 = data["images"][img3]
# print(image0.shape)

img = util_gau.convert_images([image0, image1, image2])
print("image shape: ", img[0,:,:,:].shape)
image0 = img[0,:,:,:]
image1 = img[1,:,:,:]
image2 = img[2,:,:,:]

# Convert image0 from torch.Size([3, 360, 640]) to [360, 640, 3]
image0 = image0.permute(1, 2, 0).numpy()
image1 = image1.permute(1, 2, 0).numpy()
image2 = image2.permute(1, 2, 0).numpy()

print(data["url"])

# tf_1_to_0 = tf_60_10
# tf_1_to_0 = np.linalg.inv(tf_1_to_0)
# print(data["timestamps"])

# print("offset dist ", np.linalg.norm(tf_1_to_0[:3, 3]))
print("ply files: ", file_paths)
gauss_0 = util_gau.load_ply(file_paths[0])
gauss_1 = util_gau.load_ply(file_paths[1])
# gauss_2 = util_gau.load_ply(file_paths[2])

def colorboost(channel_idx, sh):
    # add_boosts = np.clip(np.max(sh[:, channel_idx]) - sh[:, channel_idx], COLOR_BOOST)
    # sub_boosts
    sh[:, channel_idx] += np.ones_like(sh[:, channel_idx])*COLOR_BOOST
    other_channels = [i for i in range(3) if i != channel_idx]
    for i in other_channels:
        sh[:, i] -= np.ones_like(sh[:, i])*COLOR_BOOST
    return sh

# Apply the transformation matrix tf_1_to_0 to gaussians1
def fuse_maps(gaussians0, gaussians1, tf_1_to_0, gt_t1, gt_t2):
    ones = np.ones((gaussians1.xyz.shape[0], 1))
    print("dist original: ", np.linalg.norm(tf_1_to_0[:3, 3]))
    # tf_1_to_0[:3, 3] = tf_1_to_0[:3, 3]
    homogeneous_coords = np.hstack([gaussians1.xyz, ones])
    transformed_coords = (tf_1_to_0 @ homogeneous_coords.T).T[:, :3]
    gaussians1.xyz = transformed_coords
    rotation_matrix = tf_1_to_0[:3, :3]

    rotation_matrix = tf_1_to_0[:3, :3]
    rotation = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)

    quaternion = rotation.as_quat()
    # Convert gaussians1.rot to scipy Rotation object
    gaussians1_rot = scipy.spatial.transform.Rotation.from_quat(gaussians1.rot)

    # Perform quaternion multiplication
    transformed_rot = (rotation * gaussians1_rot).as_quat()
    gaussians1.rot = transformed_rot

    if COLORED:
        sh0 = colorboost(0, gaussians0.sh)
        sh1 = colorboost(1, gaussians1.sh)
    else:
        sh0 = gaussians0.sh
        sh1 = gaussians1.sh
    if SCALED:
        fused_xyz = np.concatenate([gaussians0.xyz, s_10* gaussians1.xyz], axis=0)
    elif GTSCALED:
        fused_xyz = np.concatenate([gaussians0.xyz, gt_t2/gt_t1* gaussians1.xyz], axis=0)
    else:
        fused_xyz = np.concatenate([gaussians0.xyz, gaussians1.xyz], axis=0)
    fused_rot = np.concatenate([gaussians0.rot, gaussians1.rot], axis=0)
    fused_scale = np.concatenate([gaussians0.scale, gaussians1.scale], axis=0)
    fused_opacity = np.concatenate([gaussians0.opacity, gaussians1.opacity], axis=0)
    fused_sh = np.concatenate([sh0, sh1], axis=0)

    # print(len(fused_xyz))
    fused_map = GaussianData(xyz=fused_xyz, rot=fused_rot, scale=fused_scale, opacity=fused_opacity, sh=fused_sh)
    return fused_map


used_tf = str(FRAMES[1]) + "_" + str(FRAMES[0])
print("used_tf: ", used_tf)
fused_map = fuse_maps(gauss_0, gauss_1, tf_dict["0_10"], t1, t2)

# tf2 = tf_dict["25_50"]
# tf2[:3, 3] = tf2[:3, 3] * t1/t2
# fused_map = fuse_maps(fused_map, gauss_2, tf2@tf_dict["0_25"], t1, t3)



# fused_map = fuse_maps(fused_map, gauss_2, tf_180_150@tf_150_120)
# fused_map = fuse_maps(fused_map, gauss_2, tf_120_60 @ tf_60_10)

filename = "fused_map_seq_" + str(ply_frame_0) + "_" + str(ply_frame_1) + "_" + str(ply_frame_2)
if COLORED:
    filename += "_colored"
if SCALED:
    filename += "_posescaled"
elif GTSCALED:
    filename += "_gtscaled"
elif NOPOSCALED:
    file_name += "_unscaled_"
filename += ".ply"
save_path = PATH + filename
util_gau.save_ply(save_path, fused_map)
print(f"Saved fused map to {save_path}")