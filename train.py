import torch
import torchvision

from torch.utils.data import DataLoader
from AVDataset import *
from AVC import *

video_path = "/home/mxkopy/Programming/"

dataset = VideoDataset('/home/mxkopy/Programming/Python/AudioVisualCorrelator/video/video.mp4')

data = DataLoader(dataset)

encoder = ImageEncoder()
decoder = ImageDecoder()

audio_encoder = AudioEncoder()

to_pil = torchvision.transforms.ToPILImage()

# training loop