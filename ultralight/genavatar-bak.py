import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
import pickle
# from unet2 import Model
# from unet_att import Model

import time
def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, default="")  
#parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--avatar_id', default='ultralight_avatar1', type=str)
args = parser.parse_args()

checkpoint = args.checkpoint
dataset_dir = args.dataset

img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")

avatar_path = f"./results/avatars/{args.avatar_id}"
full_imgs_path = f"{avatar_path}/full_imgs" 
face_imgs_path = f"{avatar_path}/face_imgs" 
coords_path = f"{avatar_path}/coords.pkl"
pth_path = f"{avatar_path}/ultralight.pth"
osmakedirs([avatar_path,full_imgs_path,face_imgs_path])

len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

step_stride = 0
img_idx = 0
coord_list = []

net = Model(6, 'hubert').cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()
for i in range(len_img):
    if img_idx>len_img - 1:
        step_stride = -1
    if img_idx<1:
        step_stride = 1
    img_idx += step_stride
    img_path = img_dir + str(img_idx)+'.jpg'
    lms_path = lms_dir + str(img_idx)+'.lms'
    
    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
    
    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    
    audio_feat = torch.zeros(1, 32, 32, 32)
    #print('audio_feat:',audio_feat.shape)
    audio_feat = audio_feat.cuda()
    img_concat_T = img_concat_T.cuda()
    #print('img_concat_T:',img_concat_T.shape)
    
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
        
    pred = pred.cpu().numpy().transpose(1,2,0)*255
    pred = np.array(pred, dtype=np.uint8)
    crop_img_ori[4:164, 4:164] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    img[ymin:ymax, xmin:xmax] = crop_img_ori

    cv2.putText(img, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
    cv2.imwrite(f"{full_imgs_path}/{img_idx:08d}.png", img)
    cv2.imwrite(f"{face_imgs_path}/{img_idx:08d}.png", crop_img)
    coord_list.append((xmin, ymin, xmin+w, ymin+h))

with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
os.system(f"cp {checkpoint} {pth_path}")

# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4