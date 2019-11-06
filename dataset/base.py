import torch
import os
from scipy.signal import stft
from .. import utils
import random
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            height, width,
            fps, frequency, fragment_len, batch_size,
            audio_dir, video_dir,
            window_len, overlap_len
    ):
        self.height = height
        self.width = width

        self.fps = fps
        self.frequency = frequency
        self.fragment_len = fragment_len

        self.dataset_size = len(list(filter(lambda x: x.endswith('.pt'), os.listdir(video_dir))))
        self.batch_size = batch_size

        self.load_order = torch.randperm(dataset_size)

        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.window_len = window_len
        self.overlap_len = overlap_len

    def __getitem__(self, index):
        # video and sound are assumed to be in corresponding directories
        video = torch.load(os.path.join(self.video_dir, '{}.pt'.format(self.load_order[index])))
        sound = torch.load(os.path.join(self.audio_dir, '{}.pt'.format(self.load_order[index])))

        video_len_sec = video.shape[0] / self.fps
        begin = random.uniform(0, video_len_sec - self.fragment_len)
        video = video[int(begin * self.fps):int(begin * self.fps) + self.fragment_len * self.fps]
        audio = audio[int(begin * self.frequency):int(begin * self.frequency) + self.fragment_len * self.frequency]

        data = stft(data, nperseg=self.window_len, noverlap=self.overlap_len
        data = data[2]
        data = np.abs(data).astype(np.float64)
        data = utils.transform(data)
        data = torch.Tensor(data)

        return video, sound

    def __len__(self):
        return self.dataset_size
