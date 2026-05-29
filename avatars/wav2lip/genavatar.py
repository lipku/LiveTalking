from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
from avatars.wav2lip import face_detection


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def generate_avatar(video_path, avatar_id, save_path='./data/avatars', img_size=96, pads=[0, 10, 0, 0], nosmooth=False, face_det_batch_size=16, progress_callback=None):
    """
    生成avatar的核心逻辑

    Args:
        video_path: 输入视频路径
        avatar_id: Avatar ID
        save_path: 保存根路径
        img_size: 缩放后的图像大小
        pads: 人脸框填充 [top, bottom, left, right]
        nosmooth: 是否禁用平滑
        face_det_batch_size: 人脸检测批处理大小
        progress_callback: 进度回调函数，接收 0-100 的整数
    """
    avatar_path = os.path.join(save_path, avatar_id)
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")

    osmakedirs([avatar_path, full_imgs_path, face_imgs_path])

    if progress_callback: progress_callback(5)

    print(f"正在处理视频: {video_path}")
    video2imgs(video_path, full_imgs_path, ext='png')

    if progress_callback: progress_callback(20)

    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    frames = read_imgs(input_img_list)

    if progress_callback: progress_callback(40)

    print('正在检测人脸...')
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size
    predictions = []

    while 1:
        predictions = []
        try:
            for i in range(0, len(frames), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(frames[i:i + batch_size])))
                if progress_callback:
                    progress = 40 + int((i + batch_size) / len(frames) * 40)
                    progress_callback(min(progress, 80))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU.')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, frames):
        if rect is None:
            rect = [0, 0, image.shape[1], image.shape[0]]

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)

    if progress_callback: progress_callback(85)

    coord_list = []
    print(f"正在保存人脸图片和坐标...")
    for idx, (rect, frame) in enumerate(zip(boxes, frames)):
        face_frame = frame[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
        resized_crop_frame = cv2.resize(face_frame, (img_size, img_size))
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append((int(rect[1]), int(rect[3]), int(rect[0]), int(rect[2])))

        if progress_callback:
            progress = 85 + int((idx + 1) / len(boxes) * 15)
            progress_callback(progress)

    print(f"写入数据到坐标文件: {coords_path}")
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)

    del detector
    if progress_callback: progress_callback(100)
    print("Avatar 生成完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--img_size', default=96, type=int)
    parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
    parser.add_argument('--save_path', default='data/avatars', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    args = parser.parse_args()

    generate_avatar(
        video_path=args.video_path,
        avatar_id=args.avatar_id,
        save_path=args.save_path,
        img_size=args.img_size,
        pads=args.pads,
        nosmooth=args.nosmooth,
        face_det_batch_size=args.face_det_batch_size
    )
