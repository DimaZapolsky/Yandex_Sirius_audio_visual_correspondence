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
        video = torch.load("train/videos/{}.pt".format(self.load_order[index]))
        sound = torch.load("train/sounds/{}.pt".format(self.load_order[index]))
        return video, sound

    def __len__(self):
        return self.dataset_size
