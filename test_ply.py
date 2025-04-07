from util_gau import *
import numpy as np
import os
from matplotlib import pyplot as plt

scannnet1_file = "/home/curdin/Downloads/scannet++_1.ply"

DATE = "25_03_28"
PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/"
PATH0 = "/home/curdin/master_thesis/outputs/"

date2 = "noposplat/25_03_11/1214f2a11a9fc1ed_0.ply"

nopopath = PATH0 + date2

file = PATH + "gaussian2.ply"

gaussians = load_ply(file)

nopo_gaussians = load_ply(nopopath)

scannet_gaussians = load_ply(scannnet1_file)

# gaussians.sh = gaussians.sh * 45
# gaussians.sh[:, 0] += np.ones_like(gaussians.sh[:, 0]) * 1.0

# angles = np.arctan(gaussians.sh[:, 1] / gaussians.sh[:, 0]) * 180 / np.pi
# for angle in angles:
#     print("angles: ", angle) 

gaussians.sh = gaussians.sh / np.linalg.norm(gaussians.sh, axis=1, keepdims=True)

diff_scannet = np.abs(scannet_gaussians.sh[:,1] - scannet_gaussians.sh[:,2])
print("mean diff scannet: ", np.mean(diff_scannet))
diff = np.abs(gaussians.sh[:,1] - gaussians.sh[:,2])
print("mean diff: ", np.mean(diff))

print(np.std(gaussians.opacity, axis=0))
print(np.std(scannet_gaussians.opacity, axis=0))
# print("max, min", np.max(gaussians.sh), np.min(gaussians.sh))
 
# print("sh scannet: ", scannet_gaussians.sh)

# print(np.mean(np.abs(gaussians.sh)))
# print(np.mean(np.abs(nopo_gaussians.sh)))
# print(np.std(gaussians.sh))
# print(np.std(nopo_gaussians.sh))

# print("mean opacity: ", np.mean(gaussians.opacity))
# print("mean nopo opacity: ", np.mean(nopo_gaussians.opacity))

# print(nopo_gaussians.o)

# gaussians.sh = np.exp(-gaussians.sh)
# print(np.std(gaussians.sh))
# print(np.mean(gaussians.scale))
# print("..........")
# print(np.mean(nopo_gaussians.scale))

save_ply(PATH + "gaussian_sh_normalized.ply", gaussians)

# gaussians = nopo_gaussians
# num_gauss = len(gaussians)

# plt.scatter(range(num_gauss), gaussians.sh[:, 0])
# plt.scatter(range(num_gauss), gaussians.sh[:, 1])
# plt.scatter(range(num_gauss), gaussians.sh[:, 2])


# # plt.scatter(nopo_gaussians.sh[:, 0])

# plt.show()

# image1_path = PATH + "image0.pt"
# image2_path = PATH + "image30.pt"

# image1 = torch.load(image1_path)
# image2 = torch.load(image2_path)

# plt.subplot(1, 2, 1)
# plt.title("frame 0")
# plt.imshow(image1[0, ...].permute(1, 2, 0).cpu().numpy())
# plt.subplot(1, 2, 2)
# plt.title("frame 30")
# plt.imshow(image2[0, ...].permute(1, 2, 0).cpu().numpy())

# plt.axis('off')
# plt.show()
# print(image1.shape)

# convert_images([image1, image2])
