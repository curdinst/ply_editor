import torch
import matplotlib.pyplot as plt
import util_gau
from util_gau import GaussianData
import numpy as np
from nopo_transforms import *
import scipy.spatial.transform

DATE = "25_03_13"

FRAMES = [10, 20, 30]
IMGPATH = "/home/curdin/master_thesis/images/"
DATASET_PATH = "/home/curdin/noposplat/NoPoSplat/datasets/re10k/test/"
DATASET_FILES = ["000000.torch",  "000001.torch",  "000002.torch"]
# KEY = "1214f2a11a9fc1ed"
KEY = "c48f19e2ffa52523"

PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/"

OUTPUT_PATH = "/home/curdin/master_thesis/outputs/" + DATE + "/gaussians/"


for file in DATASET_FILES:
    dataset_filepath = DATASET_PATH + file
    dataset = torch.load(dataset_filepath)
    keys = [dataset[i]["key"] for i in range(len(dataset))]
    if KEY in keys:
        print(f"Found key {KEY} in {file} at idx {keys.index(KEY)}")
        idx = keys.index(KEY)
        data = dataset[idx]
        # print("key at index 3: ", keys[3])
        break

def get_transforms(frames, key):
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
    tf_1 = data["cameras"][frame0][6:].reshape(3,4)
    tf_2 = data["cameras"][frame1][6:].reshape(3,4)
    tf_1 = torch.vstack([tf_1, torch.tensor([0, 0, 0, 1.0])])
    tf_2 = torch.vstack([tf_2, torch.tensor([0, 0, 0, 1.0])])
    tf_1_to_0 = torch.linalg.inv(tf_1) @ tf_2
    return tf_1_to_0

# print(get_transforms(["0", "10"]))

# print(data.keys())
# transfomrs = data["cameras"]
# print(data["url"])
# print("num frames: ", len(transfomrs))

# frame0 = transfomrs[FRAMES[0]][6:].reshape(3,4)
# frame1 = transfomrs[FRAMES[1]][6:].reshape(3,4)
# frame2 = transfomrs[FRAMES[2]][6:].reshape(3,4)
# frame0 = torch.vstack([frame0, torch.tensor([0, 0, 0, 1])])
# frame1 = torch.vstack([frame1, torch.tensor([0, 0, 0, 1])])
# frame2 = torch.vstack([frame2, torch.tensor([0, 0, 0, 1])])

# print(frame0)
# print(frame1)
# tf_10 = np.linalg.inv(frame1) @ frame0
# tf_21 = np.linalg.inv(frame2) @ frame1

# t1 = np.linalg.norm(tf_10[:3, 3])
# t2 = np.linalg.norm(tf_21[:3, 3])
# print("norm 1", t1)
# print("norm 2", t2)

[img1, img2, img3] = FRAMES

# img1, img2, img3 = 10, 60, 120
# img1, img2, img3 = 60, 120, 180
# img1, img2, img3 = 90, 120, 150
# img4, img5 = 280, 240


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
# Plot the images
plt.subplot(131)
plt.imshow(image0)
plt.title("Input frame " + str(img1))

plt.subplot(132)
plt.imshow(image1)
plt.title("Input frame " + str(img2))

plt.subplot(133)
plt.imshow(image2)
plt.title("Input frame " + str(img3))

plt.gcf().set_size_inches(18, 6)
plt.savefig(IMGPATH + "Input_frames_seq_" + str(img1) + "-" + str(img2) + "-" + str(img3) + ".png")
plt.show()

# Calculate the transformation matrix from frame0 to frame1
tf_1_to_0 = torch.linalg.inv(frame1) @ frame0
print("Transformation matrix from frame1 to frame0:")
print(tf_1_to_0)
# Convert tf_1_to_0 to numpy array
GT_tf_1_to_0 = tf_1_to_0.numpy()
# Plot the images
plt.subplot(131)
plt.imshow(image0)
plt.title("Input frame " + str(img1))

plt.subplot(132)
plt.imshow(image1)
plt.title("Input frame " + str(img2))

plt.subplot(133)
plt.imshow(image2)
plt.title("Input frame " + str(img3))

plt.gcf().set_size_inches(18, 6)
# plt.savefig(IMGPATH + "Input_frames_seq_" + str(img1) + "-" + str(img2) + "-" + str(img3) + ".png")
plt.show()

# Calculate the transformation matrix from frame0 to frame1
# tf_1_to_0 = torch.linalg.inv(frame1) @ frame0
# print("Transformation matrix from frame1 to frame0:")
# print(tf_1_to_0)
# # Convert tf_1_to_0 to numpy array
# GT_tf_1_to_0 = tf_1_to_0.numpy()