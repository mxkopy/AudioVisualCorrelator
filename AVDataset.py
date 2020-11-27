import torch
import torchvision
import matplotlib.pyplot as plt
import sys

from torch.utils.data import Dataset

# Contains the classes for the datasets that will be loaded in main.py

torchvision.set_video_backend('video_reader')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoDataset(Dataset):


    def __init__(self, path, store_data=False):

        self.curr_index = -1

        self.video_reader = torchvision.io.VideoReader(path, 'video')
        self.audio_reader = torchvision.io.VideoReader(path, 'audio')

        self.info = self.video_reader.get_metadata()
        self.len = int(self.info['video']['duration'][0] * self.info['video']['fps'][0])

        self.data = []
        self.store_data = store_data

        self.transform = lambda x : (x.to(device) - 127) / 255

    def __getitem__(self, index):

        self.data = []

        video_data = next(self.video_reader)['data'].type(torch.float32)
        audio_data = next(self.audio_reader)['data'].type(torch.float32)

        self.curr_index += 1

        while index > self.curr_index:

            video_data = next(self.video_reader)['data']
            audio_data = next(self.audio_reader)['data']
            self.curr_index += 1

            if self.store_data:

                self.data.append((video_data, audio_data))

        if index < self.curr_index:

            if not self.store_data:

                self.video_reader.seek(0)
                self.audio_reader.seek(0)

                self.curr_index = -1

                return self.__getitem__(index)

        return self.transform(video_data), audio_data


    def __len__(self):

        return self.len

