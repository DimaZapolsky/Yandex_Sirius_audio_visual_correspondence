import torch
import os
from scipy.signal import stft
from . import utils
import random
import numpy as np
from torchvision import transforms


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

    def normalize_video(self, video, mean, std):
        video *= (std / torch.std(video, [1, 2]))[:, None, None, :]
        video -= (torch.mean(video, [1, 2]) - mean)[:, None, None, :]
        return video

    def normalize_video_2(self, video, **kwargs):
        return video / 255

    def get_one_item(self, index):
        # video and sound are assumed to be in corresponding directories
        video = torch.load(os.path.join(self.video_dir, '{}.pt'.format(self.load_order[index]))).type(torch.Tensor)
        audio = torch.load(os.path.join(self.audio_dir, '{}.pt'.format(self.load_order[index]))).type(torch.Tensor)

        video_len_sec = video.shape[0] / self.fps
        begin = random.uniform(0, video_len_sec - self.fragment_len)
        video = video[int(begin * self.fps):int(begin * self.fps) + int(self.fragment_len * self.fps)]
        audio = audio[int(begin * self.frequency):int(begin * self.frequency) + int(self.fragment_len * self.frequency)]
        if audio.shape[0] != int(self.fragment_len * self.frequency) or video.shape[0] != int(self.fragment_len * self.fps):
            return self.get_one_item(random.randint(0, self.dataset_size - 1))

        # for i in range(video.shape[0]):
        #     video[i] = video[i].permute([2, 0, 1])
        #     torch.nn.functional.normalize(video[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     video[i] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(video[i])

        return self.normalize_video_2(video, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])), audio

    def __getitem__(self, index):
        videos = []
        audios = []
        w_sum = None

        for i in range(self.n_fragments):
            video, wave = self.get_one_item((index + i) % self.dataset_size)
            videos.append(video)
            audios.append(self.get_sg(wave / self.n_fragments))
            w_sum = wave if w_sum is None else w_sum + wave

        return torch.stack(videos), torch.stack(audios), self.get_sg(w_sum / self.n_fragments)

    def __len__(self):
        return self.dataset_size
