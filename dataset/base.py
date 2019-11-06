import math
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            height, width,
            fps, frequency, length,
            dataset_size, batch_size,
    ):
        self.height = height
        self.width = width

        self.fps = fps
        self.frequency = frequency
        self.length = length

        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.load_order = torch.randperm(dataset_size)

    def __getitem__(self, index):
        # video and sound are assumed to be in corresponding directories
        video = torch.load("video/{}.pt".format(self.load_order[index]))
        sound = torch.load("sound/{}.pt".format(self.load_order[index]))
        return video, sound

    def __iter__(self):
        for batch_index in range(math.ceil(self.dataset_size / self.batch_size)):
            batch = []
            item_index = batch_index * self.batch_size

            for _ in range(self.batch_size):
                batch.append(self.__getitem__(item_index % self.dataset_size))
                item_index += 1

            yield batch

    def __len__(self):
        return self.dataset_size
