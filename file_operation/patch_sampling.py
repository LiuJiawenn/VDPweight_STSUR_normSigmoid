import numpy as np
from numpy.lib.stride_tricks import as_strided
from file_operation.read_mat import read_mat
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import torch
import matplotlib.pyplot as plt


def extract_patches(arr, patch_shape=(32, 32, 3), extraction_step=(32, 32, 3)):
    # input (1080,1920,3)
    # 对应维度+1需要跳过的字节数
    patch_strides = arr.strides
    # 四个维度，每个维度创建一个切片
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    # patch_indices_shape = (10,33,60,1)
    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    patches = np.transpose(patches.reshape((-1, patch_shape[0], patch_shape[1], patch_shape[2])), (0, 3, 1, 2))
    return patches


def source_optical_flow_patch_extract(video_path, keyframes):
    cap = cv2.VideoCapture(video_path)
    s_patches = []
    o_patches = []

    for k in keyframes:
        if k == 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k-2)

        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        s_patch = extract_patches(frame2, (32, 32, 3), (32, 32, 3))
        o_patch = extract_patches(mag.reshape(1080, 1920, 1), (32, 32, 1), (32, 32, 1))
        s_patches.append(s_patch)
        o_patches.append(o_patch)

    cap.release()
    return s_patches, o_patches


def compressed_patch_extract(video_path, keyframe):
    cap = cv2.VideoCapture(video_path)
    patches = []
    for k in keyframe:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k-1)
        ret, frame = cap.read()
        patch = extract_patches(frame)
        patches.append(patch)
    cap.release()
    return patches


def vdp_patches(path, name):
    patches = []

    matric = np.transpose(read_mat(path, name), (3, 0, 1, 2))
    for frame in matric:
        patch = extract_patches(frame,(32,32,3),(32,32,3))
        patches.append(patch)
    return patches


def key_patch_select(s_patches, c_patches, patch_per_frame, patch_rate):
    # 通常情况下 s_patches应该是 12×1980×3×32×32 的patch列表
    # 选关键patch
    frame_nb = len(s_patches)
    patch_nb = len(s_patches[0])
    # psnr_values = np.zeros(num_patches)
    key_patch_list = []
    ratio = int(patch_rate * patch_nb)

    for i in range(frame_nb):
        psnr_temp = []

        for j in range(patch_nb):
            psnr_temp.append(psnr(s_patches[i][j], c_patches[i][j]))

        sorted_indices = np.argsort(psnr_temp)
        index_mask = np.random.choice(ratio, patch_per_frame, replace=False)
        sorted_indices = sorted_indices[index_mask]
        sorted_indices.sort()
        key_patch_list.append(sorted_indices)

    return key_patch_list


def key_patch_mask(s_patches,c_patches,so_patches,v_patches,key_patch_index):
    key_s_patches = []
    key_c_patches = []
    key_so_patches = []
    key_v_patches = []

    for i in range(len(key_patch_index)):
        key_list = key_patch_index[i]
        key_s_patches.append(s_patches[i][key_list])
        key_c_patches.append(c_patches[i][key_list])
        key_so_patches.append(so_patches[i][key_list])
        key_v_patches.append(v_patches[i][key_list])

    key_s_patches = np.concatenate(key_s_patches)
    key_c_patches = np.concatenate(key_c_patches)
    key_so_patches = np.concatenate(key_so_patches)
    key_v_patches = np.concatenate(key_v_patches)

    # plt.imshow(np.transpose(key_s_patches[12],(1,2,0)))
    # plt.imshow(np.transpose(key_c_patches[12], (1, 2, 0)))
    # plt.imshow(np.transpose(key_so_patches[12], (1, 2, 0)))
    # plt.imshow(np.transpose(key_v_patches[12], (1, 2, 0)))
    return torch.from_numpy(key_s_patches), torch.from_numpy(key_c_patches), torch.from_numpy(key_so_patches), torch.from_numpy(key_v_patches)