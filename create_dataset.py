import argparse
import torch
import torchvision
import cv2
import wave
from scipy.io import wavfile
import numpy as np
from PIL import Image
from scipy.signal import stft


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', required=True)
    parser.add_argument('--width', required=True)
    parser.add_argument('--height', required=True)
    parser.add_argument('--freq', required=True)
    parser.add_argument('--dev-size', required=True)
    parser.add_argument('--eval-size', required=True)
    parser.add_argument('--src-dir', default='./')
    parser.add_argument('--dst-dir', default='./')
    parser.add_argument('--window-len', default=1024)
    parser.add_argument('--overlap-len', default=256)
    args = parser.parse_args()
    return args


def create_dataset_videos(args):
    src_path = os.path.join(args.src_dir, '/videos/')
    dst_path = os.path.join(args.dst_dir, '/videos/')
    os.mkdirs(dst_path, exist_ok=True)
    for file in os.listdir(src_path):
        capture = cv2.VideoCapture(os.path.join(src_path, file))
        capture.set(cv2.CAP_PROP_FPS, atgs.fps)
        read_flag, frame = capture.read()
        image = Image.fromarray(frame, 'RGB')
        vid_frames = [np.asarray(torchvision.transforms.CenterCrop((args.height, args.width))(image))]

        while read_flag:
            image = Image.fromarray(frame, 'RGB')
            vid_frames.append(np.asarray(torchvision.transforms.CenterCrop((args.height, args.width))(image)))

            read_flag, frame = capture.read()

        vid_frames = np.asarray(vid_frames, dtype='uint8')
        vid_frames = tf.Tensor(vid_frames)
        capture.release()
        torch.save(vid_frames, os.path.join(dst_path, file.split('.')[0] + '.pt'))


def create_dataset_audios(args):
    src_path = os.path.join(args.src_dir, '/audios/')
    dst_path = os.path.join(args.dst_dir, '/audios/')
    os.mkdirs(dst_path, exist_ok=True)
    for file in os.listdir(src_path):
        fs, data = wavfile.read(os.path.join(src_path, file))
        data = stft(data, nperseg=args.window_len, noverlap=args.overlap_len)
        data = tf.Tensor(data)
        torch.save(data, os.path.join(dst_path, file.split('.')[0] + '.pt'))


def main():
    args = parse_args()
    create_dataset(args)


if __name__ == '__main__':
    main()