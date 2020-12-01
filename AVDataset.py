import torch
import torchvision
import torchaudio
import torchaudio.backend.sox_io_backend as sox_io
import matplotlib.pyplot as plt
import sys
import os

from torch.utils.data import Dataset

# Contains the classes for the datasets that will be loaded in main.py

torchvision.set_video_backend('pyav')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioVisualDataset(Dataset):


    # Initializes the dataset assuming path leads to .mp4 file
    # store_data determines if frames are cached for later use
    # streams determines whether to include audio or video
        # 0 - include video
        # 1 - include audio
        # 2 - include video and audio (default)

    def __init__(self, path, streams=2):

        # Having this helper index makes everything go incredibly fast
        self.curr_index = -1

        # To ensure playback and whatnot we reencode to just audio using the native ffmpeg

        os.system(f'ffmpeg -n -i {path} -acodec pcm_s16le -ar 44100 {path}.wav'.format(path))

        # Video loading is very simple

        self.video_reader = torchvision.io.VideoReader(path, 'video')
        self.visual_info = self.video_reader.get_metadata()['video']

        print(self.visual_info)

        # Since in general, video fps <<< audio fps, we want to sync the audio to the video. 
        # So the audio initialization is more complicated.

        # Recall that samples/second * (second/frame) = samples/frame
        self.audio_info = {
            'sample_rate' : sox_io.info(path + '.wav').sample_rate, 
            'num_frames' : sox_io.info(path + '.wav').num_frames, 
            'num_channels' : sox_io.info(path + '.wav').num_channels }

        self.a_v_ratio = int(self.audio_info['sample_rate'] / self.visual_info['fps'][0])
        self.audio_reader = lambda index: sox_io.load(path + '.wav', index * self.a_v_ratio, self.a_v_ratio, normalize=True)

        # Wrapper to make the iteration much more simple
        if streams == 0:
            self.streamer = lambda _ : next(self.video_reader)['data'].type(torch.float32).to(device)

        if streams == 1:
            self.streamer = lambda index: self.audio_reader(index)[0].type(torch.float32).to(device)

        if streams == 2:
            self.streamer = lambda index: self.audio_reader(index)[0].type(torch.float32).to(device), next(self.video_reader)['data'].type(torch.float32).to(device)


    # The output elements will have shape
    #(C1, H, W)   ,    (C2, L)   or    ((C1, H, W), (C2, F))

    # C1 and C2 are video and audio channels (i.e. rgb, stereo)
    # L is the length of the audio frame (i.e. num_audio_frames_read)

    def __getitem__(self, index):

        self.curr_index += 1

        if index == self.curr_index:
            return self.streamer(index)

        elif index < self.curr_index:
    
            self.curr_index = -1
            self.video_reader.seek(0.0)

        return self.__getitem__(index)


        # self.video_reader.seek(index / self.visual_info['fps'][0])

        # return self.streamer(index)


    def __len__(self):

        return int(self.visual_info['duration'][0] * self.visual_info['fps'][0])
        
