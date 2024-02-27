import os
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from file_operation.patch_sampling import vdp_patches, source_optical_flow_patch_extract, compressed_patch_extract, key_patch_select, \
    key_patch_mask


def h264_to_avi(path_264, path_avi):
    # print("s_264  ", path_264)
    # print("s_avi: ", path_avi)
    if os.path.exists(path_avi):
        return

    stream = ffmpeg.input(path_264)
    stream = ffmpeg.output(stream, path_avi, vcodec='rawvideo')
    ffmpeg.run(stream)


def data_generator(patch_per_frame=64, patch_rate=0.4, key_frame_nb=12, shuffle=True,videoList=np.array(list(range(44, 220)))):
    # TODO：
    # 1. 按keyframe读取帧转换为patch
    # 2. 生成optical 转换为patch
    # 3. 读取VDP
    # 4. 准备SURlabel
    raw_path = 'F:/264videos/'
    running_folder = 'C:/Users/AAAAA/Desktop/runningfolder/'
    vdp_path = 'F:/VDP/'
    kf_list = np.load('data/kf12_list.npy')
    SUR_GROUND_TRUTH = np.load("data/SUR_GROUND_TRUTH.npy")
    trainVideos = videoList
    # trainVideos = np.load('data/trainsetVideos.npy')
    if shuffle:
        np.random.shuffle(trainVideos)

    tag = str(int((patch_per_frame+patch_rate)*10))

    dir_list = os.listdir(raw_path)
    # 45-176是训练集,就是video index
    # for i in range(44, 176):
    # 改成随机生成的视频序列,且每个epoch随机打乱
    for i in trainVideos:
        current_video = dir_list[i]
        s_264 = raw_path + current_video + '/' + current_video + '_qp_00.264'
        s_avi = running_folder + current_video + '_qp_00.avi'
        h264_to_avi(s_264, s_avi)
        s_patches, o_patches = source_optical_flow_patch_extract(s_avi, kf_list[i])

        for qp in [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 51]:
            c_264 = raw_path + current_video + '/' + current_video + '_qp_'+'{:02d}'.format(qp)+'.264'
            c_avi = running_folder + current_video + tag + '_qp_'+'{:02d}'.format(qp)+'.avi'
            h264_to_avi(c_264, c_avi)
            c_patches = compressed_patch_extract(c_avi, kf_list[i])

            v_mat = vdp_path + current_video+'/'+current_video + '_qp_'+'{:02d}'.format(qp) + '.mat'
            v_patches = vdp_patches(v_mat,'diff_map_kf')

            key_patch_index = key_patch_select(s_patches, c_patches, patch_per_frame, patch_rate)
            key_s_patches, key_c_patches, key_o_patches, key_v_patches = key_patch_mask(s_patches, c_patches,
                                                                                          o_patches, v_patches,
                                                                                          key_patch_index)
            yield (key_s_patches, key_c_patches, key_o_patches, key_v_patches), SUR_GROUND_TRUTH[i][qp - 1]

            # 删除上一个压缩视频
            os.remove(c_avi)

        # 删除上一个原视频
        os.remove(s_avi)