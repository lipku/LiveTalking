import argparse
from os import wait3

import numpy as np
import cv2
import math

import torch
import torchvision
from .detect_face import SCRFD
# from models.pfld_lite import PFLDInference
# from models.pfld import PFLDInference
from .pfld_mobileone import PFLD_GhostOne as PFLDInference
def face_det(img, model):

    cropped_imgs = []
    boxes_list = []
    center_list = []
    alpha_list = []
    height, width = img.shape[:2]
    bboxes, indices, kps = model.detect(img)

    for i in indices:
        x1, y1, x2, y2 = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(bboxes[i, 1] + bboxes[i, 3])
        p1 = kps[i,0]
        p2 = kps[i,1]
        w = x2 - x1
        h = y2 - y1
        cx = (x2+x1)//2
        cy = (y2+y1)//2
        wh = np.asarray([w,h])
        boxsize = int(np.max(wh)*1.05)
        
        size = boxsize
        xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
        x1, y1 = xy
        x2, y2 = xy + size
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        cropped = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx >0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            y1 = y1-dy
            x1 = x1-dx
        center = (int((x2-x1)//2), int((y2-y1)//2))
        
        boxes_list.append([x1,y1,x2,y2])
        center_list.append(center)
        alpha = math.atan2(p2[1]-p1[1], p2[0]-p1[0]) * 180 / math.pi
        rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
        # img_rotated_by_alpha = cv2.warpAffine(cropped, rot_mat,
        #                                     (cropped.shape[1], cropped.shape[0]))

        # cropped_imgs.append(img_rotated_by_alpha)
        cropped_imgs.append(cropped)
        
        alpha_list.append(alpha)
        
        break
    return cropped_imgs, boxes_list, center_list, alpha_list

class Landmark:
    def __init__(self):
        
        with open('./face_detect_utils/mean_face.txt', 'r') as f_mean_face:
            mean_face = f_mean_face.read()
        self.mean_face = np.asarray(mean_face.split(' '), dtype=np.float32)
        self.det_net = SCRFD('./face_detect_utils/scrfd_2.5g_kps.onnx', confThreshold=0.1, nmsThreshold=0.5)

        checkpoint = torch.load('./face_detect_utils/checkpoint_epoch_335.pth.tar')
        self.pfld_backbone = PFLDInference().cuda()
        self.pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
        self.pfld_backbone.eval()

    def detect(self, img_path):
        
        img = cv2.imread(img_path)
        img_ori = img.copy()

        h,w = img_ori.shape[:2]
        cropped_imgs, boxes_list, center_list, alpha_list = face_det(img, self.det_net)
        cropped = cropped_imgs[0]
        # cv2.imshow("cropped", cropped)
        h,w = cropped.shape[:2]
        x1, y1, x2, y2 = boxes_list[0]
        transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])
        input = cv2.resize(cropped, (192, 192))
        input = np.asarray(input, dtype=np.float32) / 255.0
        input = input.transpose(2,0,1)
        input = torch.from_numpy(input)[None]
        input = input.cuda()
        # print(input)
        # asd

        # input = transform(input).unsqueeze(0).cuda()
        landmarks = self.pfld_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy()
        pre_landmark = pre_landmark + self.mean_face

        pre_landmark = pre_landmark.reshape(-1, 2)
        pre_landmark[:,0] *= w
        pre_landmark[:,1] *= h
        pre_landmark = pre_landmark.astype(np.int32)
        return pre_landmark, x1, y1
        