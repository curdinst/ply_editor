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
from pathlib import Path
# from icp import icp
import open3d as o3d
from scipy.spatial import procrustes
from nopo_transforms import *
from scale_estimation import *
from scale_aligning_icp import *
# from show_imgs import *

SEQUENCE = [0, 20, 10]

COLORED = True
SCALED = True
GTSCALED = False
NOPOSCALED = False
assert (SCALED and GTSCALED) == False

# DATE = "25_03_13_scale_inv_false"
# DATE = "25_03_19"
# DATE = "25_03_20_mb1False_relposeFalse"
# DATE = "25_03_21"
DATE = "25_03_21_re10kdl3dv"

# FRAMES = [0, 10, 20]
COLOR_BOOST = 0.8
IMGPATH = "/home/curdin/master_thesis/images/"
DATASET_PATH = "/home/curdin/noposplat/NoPoSplat/datasets/re10k/test/"
DATASET_FILES = ["000000.torch",  "000001.torch",  "000002.torch"]
# KEY = "1214f2a11a9fc1ed"
KEY = "c48f19e2ffa52523"

PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/"

noposcaled = "_unscaled_" if NOPOSCALED else "_"

OUTPUT_PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/gaussians/"

# file0 = KEY + noposcaled + str(FRAMES[0]) + "_" + str(FRAMES[1]) + ".ply"
# file1 = KEY + noposcaled + str(FRAMES[1]) + "_" + str(FRAMES[2]) + ".ply"
# file2 = KEY + noposcaled + str(FRAMES[2]) + "_" + str(FRAMES[3]) + ".ply"

# print("Files: ", file0, "\n", file1)

# Load transformation matrices from JSON file
tf_dict = {}
tf_scaled_dict = {}
poses_path = PATH + 'poses/'

for file_name in os.listdir(poses_path):
    if file_name.endswith('.json') and KEY in file_name:
        # print(file_name)
        index0 = file_name.split('_')[1]
        index1 = file_name.split('_')[2].split('.')[0]
        with open(os.path.join(poses_path, file_name), 'r') as f:
            tf_dict[index0 + "_" + index1] = np.array(json.load(f)).squeeze()
    elif file_name.endswith('.pt') and KEY in file_name:
        tf = torch.load(os.path.join(poses_path, file_name)).cpu()
        index0 = file_name.split('_')[1]
        index1 = file_name.split('_')[2].split('.')[0]
        tf_dict[index0 + "_" + index1] = torch.Tensor.numpy(tf[0,0])


# print(tf_dict["0_10"])
# tf_12 = tf_dict["0_10"]
# r_21 = tf_12[:3, :3].T
# t_21 = -tf_12[:3, :3].T @ tf_12[:3, 3]
# tf_21 = np.eye(4)
# tf_21[:3, :3] = r_21
# tf_21[:3, 3] = t_21
# print("tf_21: \n", tf_21)
# 1/0
target_poses_tensor = None
tf_target_dict = {}
tf_input_target_dict = {}
target_poses_path = PATH + 'target_poses/'
for file_name in os.listdir(target_poses_path):
    if file_name.endswith('.pt') and KEY in file_name:
        # print(file_name)
        # print("key in file: ", )
        index0 = file_name.split('_')[1]
        index1 = file_name.split('_')[2]
        path = os.path.join(target_poses_path, file_name)
        target_poses = torch.load(path)
        target_poses_tensor = target_poses
        target_poses = target_poses.tolist()[0]
        tf_input_target_dict[index0 + "_" + index1] = np.array(target_poses[1]).squeeze()
        step = int(index1) - int(index0)
        tf_target_dict[index0 + "_" + str(int(index1)+step)] = np.array(target_poses[0]).squeeze()
        if int(index0) == 0: continue
        # print("path: ", path)
        # print(target_poses)
        # target_poses = target_poses.tolist()[0]
        tf_target_dict[index0 + "_" + str(int(index0)-step)] = np.array(target_poses[0]).squeeze()


# file_paths = [OUTPUT_PATH + file for file in files]
# print(target_poses_tensor)
# r_01 = target_poses_tensor[0,0,:3,:3]
# t_01 = target_poses_tensor[0,0,:3,3]
# tf_10 = torch.eye(4)
# tf_10[:3,:3] = r_01.T
# tf_10[:3,3] = -r_01.T @ t_01
# target_poses_tensor[0,0] = tf_10
# print(target_poses_tensor)
# 1/0

def get_gt_tfs():
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
    return t1, t2

def get_gt_transforms(frames, key):
    for file in DATASET_FILES:
        dataset_filepath = DATASET_PATH + file
        dataset = torch.load(dataset_filepath)
        keys = [dataset[i]["key"] for i in range(len(dataset))]
        if key in keys:
            # print(f"Found key {key} in {file} at idx {keys.index(KEY)}")
            idx = keys.index(key)
            data = dataset[idx]
            # print("key at index 3: ", keys[3])
            break
    frame0 = int(frames[0])
    frame1 = int(frames[1])
    frame2 = int(frames[2])
    tf_1 = data["cameras"][frame0][6:].reshape(3,4)
    tf_2 = data["cameras"][frame1][6:].reshape(3,4)
    tf_3 = data["cameras"][frame2][6:].reshape(3,4)
    tf_1 = torch.vstack([tf_1, torch.tensor([0, 0, 0, 1.0])])
    tf_2 = torch.vstack([tf_2, torch.tensor([0, 0, 0, 1.0])])
    tf_3 = torch.vstack([tf_3, torch.tensor([0, 0, 0, 1.0])])
    tf_1_to_0 = torch.linalg.inv(tf_1) @ tf_2
    scale = np.linalg.norm(tf_3[:3, 3] - tf_2[:3, 3]) / np.linalg.norm((tf_2[:3, 3] - tf_1[:3, 3]))
    return tf_1_to_0, scale

# print(data["url"])

def load_ply_file(frames):
    print("loading ply file: ", frames)
    file = KEY + noposcaled + str(frames[0]) + "_" + str(frames[1]) + ".ply"
    file_path = OUTPUT_PATH + file
    return util_gau.load_ply(file_path)

def colorboost(channel_idx, sh):
    # sub_boosts
    sh[:, channel_idx] += np.ones_like(sh[:, channel_idx])*COLOR_BOOST
    other_channels = [i for i in range(3) if i != channel_idx]
    for i in other_channels:
        sh[:, i] -= np.ones_like(sh[:, i])*COLOR_BOOST
    return sh

def scale_gaussians(gaussians, scale):
    gaussians.xyz *= scale
    gaussians.scale *= scale
    return gaussians

def run_icp(src, target, init_tf):
    target = load_ply_file([0, 10])
    src = load_ply_file([10, 20])

    src.xyz *= np.linalg.norm(np.mean(target.xyz, axis=0))/np.linalg.norm(np.mean(src.xyz, axis=0))
    opacity_threshold = 0.6
    # print("median opacity: ", np.median(src.opacity))
    # src_mask = src.opacity > opacity_threshold
    # target_mask = target.opacity > opacity_threshold
    # print(np.count_nonzero(src_mask), np.count_nonzero(target_mask))
    # print(src.xyz.shape)
    
    # src_points = src.xyz[src_mask.squeeze(), :]
    # target_points = target.xyz[target_mask.squeeze(), :]
    src_points = src.xyz
    target_points = target.xyz
    init_tf = np.eye(4)
    print("src_points: ", src.xyz.shape, src_points.shape)
    print("target_points: ", target.xyz.shape, target_points.shape)
    aligned_source, transformation, scale = scale_aligning_icp(src_points, target_points, init_tf)
    src.xyz *= scale
    fused_map = fuse_maps(src, target, transformation, 0, 0, 0)
    save_path = Path(PATH + KEY + "/fused_map_icp.ply")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    util_gau.save_ply(save_path, fused_map)
    print(f"Saved fused map to {save_path}")

    return aligned_source, transformation, scale


# Apply the transformation matrix tf_1_to_0 to gaussians1
def fuse_maps(gaussians0, gaussians1, tf_1_to_0, color_channel = None, gt_t1 = None, gt_t2 = None):
    ones = np.ones((gaussians1.xyz.shape[0], 1))
    # print("dist original: ", np.linalg.norm(tf_1_to_0[:3, 3]))
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

    sh0 = gaussians0.sh
    if color_channel is not None and COLORED:
        sh1 = colorboost(color_channel, gaussians1.sh)
    else:
        sh1 = gaussians1.sh
    if SCALED:
        fused_xyz = np.concatenate([gaussians0.xyz, gaussians1.xyz], axis=0)
    # elif GTSCALED:
    #     fused_xyz = np.concatenate([gaussians0.xyz, gt_t2/gt_t1* gaussians1.xyz], axis=0)
    else:
        fused_xyz = np.concatenate([gaussians0.xyz, gaussians1.xyz], axis=0)
    fused_rot = np.concatenate([gaussians0.rot, gaussians1.rot], axis=0)
    fused_scale = np.concatenate([gaussians0.scale, gaussians1.scale], axis=0)
    fused_opacity = np.concatenate([gaussians0.opacity, gaussians1.opacity], axis=0)
    fused_sh = np.concatenate([sh0, sh1], axis=0)

    # print(len(fused_xyz))
    fused_map = GaussianData(xyz=fused_xyz, rot=fused_rot, scale=fused_scale, opacity=fused_opacity, sh=fused_sh)
    return fused_map

def fuse_sequence(start_frame, end_frame, step):
    fused_map = load_ply_file((start_frame, start_frame + step)) # leave first gaussians unscaled
    last_mean = np.linalg.norm(np.mean(fused_map.xyz, axis=0))
    print("mean: ", last_mean)
    tf_scaled_dict[str(start_frame) + "_" + str(start_frame+step)] = tf_dict[str(start_frame) + "_" + str(start_frame+step)] # first tf is in scale
    colered, scaled = "", ""
    if COLORED:
        colered = "color_"
    if not SCALED:
        scaled = "unscaled_"
        if GTSCALED:
            scaled = "gtscaled_"
    filename = "fused_map_seq_icp" + scaled + colered + str(start_frame) + "_" + str(end_frame) + "_" + str(step) + ".ply"
    i = 0
    tf_scaled = [tf_dict[str(start_frame) + "_" + str(start_frame+step)]]
    for idx in range(start_frame, end_frame, step):
        # print("GT TF:", np.linalg.norm(gt_tf[:3, 3]))
        if idx == start_frame:
            continue
        gt_tf, gt_scale = get_gt_transforms([str(idx-step), str(idx), str(idx+step)], KEY)

        gaussians = load_ply_file((idx, idx + step))
        tf_01_key = str(idx-step) + "_" + str(idx)
        tf_10_key = str(idx) + "_" + str(idx-step)
        tf_12_key = str(idx) + "_" + str(idx+step)
        tf_02_key = str(idx-step) + "_" + str(idx+step)
        
        print("=========================================================")
        print("gt tf: \n", gt_tf)
        print("tf 01 from pose_eval: \n", tf_dict[tf_01_key])
        print("----------------------------------------------------------")
        print("gt_t1: ", gt_tf[:3, 3])
        error_r_01 = gt_tf[:3, :3] @ tf_dict[tf_01_key][:3, :3]
        print(tf_dict[tf_01_key][:3, 3])
        trace_value = np.trace(error_r_01)
        theta = np.arccos((trace_value - 1) / 2)
        print("theta r01: ", theta)
        # error_t_01 = gt_tf[:3, 3] - tf_dict[tf_01_key][:3, 3]
        # print("error t01: ", np.linalg.norm(error_t_01))
        
        error_r_10 = gt_tf[:3, :3] @ tf_target_dict[tf_10_key][:3, :3].T
        trace_value10 = np.trace(error_r_10)
        theta10 = np.arccos((trace_value10 - 1) / 2)
        print("theta r10: ", theta10)

        print("keys: ", tf_01_key, tf_10_key)
        scale = estimate_scale(tf_01=tf_scaled_dict[tf_01_key], tf_10=tf_target_dict[tf_10_key], tf_02 = tf_target_dict[tf_02_key], tf_12 = tf_dict[tf_12_key])
        mean = np.linalg.norm(np.mean(gaussians.xyz, axis=0))
        # print("mean: ", mean)
        means_scale = last_mean/mean
        last_mean = mean*scale
        if GTSCALED: scale = gt_scale
        
        print("scale: ", scale, "means scale: ", means_scale, "gt scale: ", gt_scale)
        if SCALED:
            # print("mean:", np.linalg.norm(np.mean(gaussians.xyz, axis=0)))
            gaussians_scaled = scale_gaussians(gaussians, scale)
            print("mean after scaling:", np.linalg.norm(np.mean(gaussians_scaled.xyz, axis=0)))
        else:
            gaussians_scaled = gaussians

        aligned_source, transformation, icp_scale = run_icp(gaussians_scaled, fused_map, tf_scaled[-1])
        print("ICP transformation, scale: ", transformation, scale)
        print(transformation.shape)
        # h_cords = np.hstack([gaussians_scaled.xyz, np.ones((gaussians_scaled.xyz.shape[0], 1))])
        # transformed_coords = (tf_scaled[-1] @ h_cords.T).T[:, :3]
        # gaussians_scaled.xyz = transformed_coords


        gaussians_scaled.xyz *= icp_scale
        fused_map = fuse_maps(fused_map, gaussians_scaled, transformation, i%3, 0, 0)
        tf_12 = tf_dict[tf_12_key]
        if SCALED:
            tf_12[:3, 3] = tf_12[:3, 3] * scale
        tf_scaled_dict[tf_12_key] = tf_12
        tf_scaled.append(tf_scaled[-1] @ tf_12)
        last_gaussians = gaussians_scaled
        i += 1
    save_path = Path(PATH + KEY + "/" + filename)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    util_gau.save_ply(save_path, fused_map)
    print(f"Saved fused map to {save_path}")


[startframe, endframe, step] = SEQUENCE
# fuse_sequence(startframe, endframe, step)

run_icp(None, None, None)


# filename = "fused_map_seq_" + str(ply_frame_0) + "_" + str(ply_frame_1) + "_" + str(ply_frame_2)
# if COLORED:
#     filename += "_colored"
# if SCALED:
#     filename += "_posescaled"
# elif GTSCALED:
#     filename += "_gtscaled"
# elif NOPOSCALED:
#     file_name += "_unscaled_"
# filename += ".ply"
# save_path = PATH + filename
# util_gau.save_ply(save_path, fused_map)
# print(f"Saved fused map to {save_path}")