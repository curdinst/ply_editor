# import torch
# import glfw
# import OpenGL.GL as gl
# from imgui.integrations.glfw import GlfwRenderer
# import imgui
# import numpy as np
# import util
# import imageio
# import torch
# import util_gau
# from util_gau import GaussianData
# import numpy as np
# import json
# import os
# import matplotlib.pyplot as plt
# import scipy.spatial.transform
# from pathlib import Path
# from icp import icp
import open3d as o3d
# from open3d import *    
# from scipy.spatial import procrustes
# from nopo_transforms import *
# from scale_estimation import *
# from scale_aligning_icp import *
# from show_imgs import *


DATE = "25_03_28"

PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/logs/"

OUTPUT_PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/gaussians/"


FILENAME = "rgbd_dataset_freiburg1_room.ply"

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import plyfile

# # Load the PLY file
# ply = plyfile.PlyData.read(PATH+FILENAME)
# x = np.array(ply['vertex']['x'])
# y = np.array(ply['vertex']['y'])
# z = np.array(ply['vertex']['z'])
# r = np.array(ply['vertex']['red']) / 255.0
# g = np.array(ply['vertex']['green']) / 255.0
# b = np.array(ply['vertex']['blue']) / 255.0

# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=np.vstack((r, g, b)).T, s=1)
# plt.show()

# Load the PLY file
pcd = o3d.io.read_point_cloud(PATH+FILENAME)

# Visualize
o3d.visualization.draw_geometries([pcd])

# cloud = io.read_point_cloud(OUTPUT_PATH+FILENAME) # Read point cloud
# visualization.draw_geometries([cloud])