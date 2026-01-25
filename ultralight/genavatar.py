import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pickle
from glob import glob

from face_detect_utils.get_landmark import Landmark
# from unet2 import Model
# from unet_att import Model

import time
def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', default='', type=str)
parser.add_argument('--img_size', default=168, type=int)
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--avatar_id', default='ultralight_avatar1', type=str)
args = parser.parse_args()

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    print(f"即将使用OpenCV将视频: {vid_path} 转换为图片")
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break
    print("视频转换完成")

def read_imgs(img_list):
    frames = []
    print('读取图片到内存...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
if __name__ == "__main__":
    avatar_path = f"./results/avatars/{args.avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    pth_path = f"{avatar_path}/ultralight.pth"
    osmakedirs([avatar_path,full_imgs_path,face_imgs_path])
    print(args)

    video2imgs(args.video_path, full_imgs_path, ext = 'png')
    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

    #frames = read_imgs(input_img_list)
    #face_det_results = face_detect(frames) 
    coord_list = []
    idx = 0
    print(f"开始人脸检测")
    landmark = Landmark()
    target_size = args.img_size
    for i in tqdm(range(len(input_img_list))):
        img = cv2.imread(input_img_list[i])
        lms, x1, y1 = landmark.detect(input_img_list[i])
        xmin = lms[1][0]+x1
        ymin = lms[52][1]+y1

        xmax = lms[31][0]+x1
        width = xmax - xmin
        ymax = ymin + width
        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (target_size, target_size), cv2.INTER_AREA)
        # cv2.imwrite(f"{full_imgs_path}/{idx:08d}.png", img)
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", crop_img)
        coord_list.append((xmin, ymin, xmin+w, ymin+h))
        idx = idx + 1

    print(f"共检测到{idx}张人脸")
	
    print(f"写入数据到坐标文件:{coords_path}")
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
    os.system(f"cp {args.checkpoint} {pth_path}")