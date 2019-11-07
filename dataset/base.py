import torch
import os
from scipy.signal import stft
from . import utils
import random
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            height, width,
            fps, frequency, fragment_len, batch_size,
            audio_dir, video_dir,
            window_len, overlap_len,
            n_fragments=2
    ):
        self.height = height
        self.width = width

        self.fps = fps
        self.frequency = frequency
        self.fragment_len = fragment_len

        self.dataset_size = len(list(filter(lambda x: x.endswith('.pt'), os.listdir(video_dir))))
        self.batch_size = batch_size

        self.load_order = torch.randperm(self.dataset_size)

        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.window_len = window_len
        self.overlap_len = overlap_len

        self.n_fragments = n_fragments

    def get_sg(self, audio):
        data = stft(audio, nperseg=self.window_len, noverlap=self.overlap_len)
        data = data[2]
        data = torch.Tensor(np.abs(data).astype(np.float64))
        data = data[:, :(data.shape[1] // 16) * 16]
        data = utils.transform(data)
        return data[None, :, :]

    def get_one_item(self, index):
        # video and sound are assumed to be in corresponding directories
        video = torch.load(os.path.join(self.video_dir, '{}.pt'.format(self.load_order[index])))
        audio = torch.load(os.path.join(self.audio_dir, '{}.pt'.format(self.load_order[index])))

        video_len_sec = video.shape[0] / self.fps
        begin = random.uniform(0, video_len_sec - self.fragment_len)
        video = video[int(begin * self.fps):int(begin * self.fps) + int(self.fragment_len * self.fps)]
        audio = audio[int(begin * self.frequency):int(begin * self.frequency) + int(self.fragment_len * self.frequency)]
        if audio.shape[0] != int(self.fragment_len * self.frequency) or video.shape[0] != int(self.fragment_len * self.fps):
            return self.get_one_item(random.randint(0, self.dataset_size - 1))

        return video, audio, self.get_sg(audio)

    def __getitem__(self, index):
        videos = []
        audios = []
        w_sum = None

        for i in range(self.n_fragments):
            video, wave, audio_sg = self.get_one_item((index + i) % self.dataset_size)
            videos.append(video)
            audios.append(audio_sg)
            w_sum = wave if w_sum is None else w_sum + wave

        return torch.stack(videos), torch.stack(audios), self.get_sg(w_sum)

    def __len__(self):
        return self.dataset_size
