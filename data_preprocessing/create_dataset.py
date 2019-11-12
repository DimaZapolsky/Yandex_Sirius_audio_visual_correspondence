import argparse
import torch
import os
import cv2
import wave
from scipy.io import wavfile
import numpy as np
from PIL import Image
from scipy.signal import stft
import time
import torchvision
import librosa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', required=True, type=int)
    parser.add_argument('--width', required=True, type=int)
    parser.add_argument('--height', required=True, type=int)
    parser.add_argument('--freq', required=True, type=int)
    parser.add_argument('--src-dir', default='./')
    parser.add_argument('--dst-dir', default='./')
    parser.add_argument('--window-len', default=1022, type=int)
    parser.add_argument('--overlap-len', default=766, type=int)
    args = parser.parse_args()
    return args


def create_dataset_videos(args, src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)

    for file in os.listdir(src_path):
        if not os.path.isfile(os.path.join(src_path, file)) or not file.endswith('.mp4'):
            continue
        start_time = time.time()
        capture = cv2.VideoCapture(os.path.join(src_path, file))
        src_fps = capture.get(cv2.CAP_PROP_FPS)
        vid_frames = []
        i = 0
        j = 0
        while True:
            read_flag, frame = capture.read()
            if not read_flag:
                break
            if i == int(j * src_fps / args.fps):
                image = Image.fromarray(frame, 'RGB')
                image = torchvision.transforms.Resize(args.width)(image)
                image = torchvision.transforms.CenterCrop((args.height, args.width))(image)
                vid_frames.append(np.asarray(image))
                j += 1
            i += 1

        vid_frames = np.asarray(vid_frames, dtype=np.uint8)
        vid_frames = torch.ByteTensor(vid_frames)
        capture.release()

        torch.save(vid_frames, os.path.join(dst_path, file.split('.')[0] + '.pt'))


def stereo_to_mono(wave):
    return (wave[:, 0] + wave[:, 1]) // 2


def create_dataset_audios(args, src_path, dst_path, silent=False):
    os.makedirs(dst_path, exist_ok=True)
    for file in os.listdir(src_path):
        if not os.path.isfile(os.path.join(src_path, file)) or not file.endswith('.wav'):
            continue
        data, fs = librosa.load(os.path.join(src_path, file), sr=args.freq)
        if len(data.shape) > 1:
            data = stereo_to_mono(data)
        data = torch.Tensor(data)
        if silent:
            data *= 0
        torch.save(data, os.path.join(dst_path, file.split('.')[0] + '.pt'))


def main():
    args = parse_args()
    create_dataset_videos(args, os.path.join(args.src_dir, 'videos/solo'), os.path.join(args.dst_dir, 'videos/solo'))
    create_dataset_videos(args, os.path.join(args.src_dir, 'videos/duet'), os.path.join(args.dst_dir, 'videos/duet'))
    create_dataset_videos(args, os.path.join(args.src_dir, 'videos/silent'), os.path.join(args.dst_dir, 'videos/silent'))
    create_dataset_audios(args, os.path.join(args.src_dir, 'audios/solo'), os.path.join(args.dst_dir, 'audios/solo'))
    create_dataset_audios(args, os.path.join(args.src_dir, 'audios/duet'), os.path.join(args.dst_dir, 'audios/duet'))
    create_dataset_audios(args, os.path.join(args.src_dir, 'audios/silent'), os.path.join(args.dst_dir, 'audios/silent'))


if __name__ == '__main__':
    main()
