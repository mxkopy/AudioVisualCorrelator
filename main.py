import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv 
import os

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

from AVDataset import *
from AVC import *

# static things, probably wont change between training loops

encoder = ImageEncoder()
decoder = ImageDecoder()

audio_encoder = AudioEncoder()
audio_decoder = AudioDecoder()

to_pil = torchvision.transforms.ToPILImage()

loss = torch.nn.MSELoss()
lr = 1e-2
optimizer = Adam(list(encoder.parameters()) + list(encoder.parameters()), lr)

project_path = '/home/mxkopy/Programming/Python/AudioVisualCorrelator'
model_path = project_path + '/models'


# dynamic, might wanna find a way to crawl through /video
video_path = project_path + '/video/video.mp4'

dataset = VideoDataset(video_path)
data = DataLoader(dataset, batch_size=4)

# find a way to plug these values in
resize = torchvision.transforms.Resize((512, 512))

encoder.train()

decoder.train()

for epoch in range(4):

    running_loss = 0.0

    for batch in data:

        optimizer.zero_grad()

        video, audio = batch

        # audio = torch.transpose(audio, 1, 2)

        img_out = decoder(encoder(video))

        truth = resize(video)

        curr_loss = loss(img_out, truth)

        running_loss += curr_loss.item()

        # running_loss += loss(img_out, truth)

        curr_loss.backward()

        optimizer.step()

        print(running_loss)

        # print(img_out)

        cv.imshow('woa', (img_out[0].detach().numpy().transpose(2, 1, 0) * 255 + 127))

        cv.waitKey(1)

cv.destroyAllWindows()
