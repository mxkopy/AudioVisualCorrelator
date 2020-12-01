import torch 
import torchvision
import numpy as np
import torchaudio
import cv2 as cv 
import os

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from PIL import Image

from AVDataset import *
from AVC import *


# Potential user input
lr = 1e-4
batch_size = 1
epochs = 1
torchaudio.set_audio_backend('sox_io')

# Initialization

project_path = os.getcwd()
model_path = project_path + '/models'

# Dataset is an iterator that returns a tuple that contains tensors

# (C, H, W) where C, H, W are rgb channels, heigh, width
# (C, L) where C is L/R channels, and L is the length of an audio frame

# It can return video, audio, or (video, audio)

dataset = AudioVisualDataset(project_path + '/video/0.mp4', streams=2)

# Data is an iterator that returns, where B is number of batches, tensors of shape
# (B, C, H, W) 
# (B, C, L)
# (B, (C, H, W), (C, L))

data = DataLoader(dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = ImageEncoder().to(device)
decoder = ImageDecoder().to(device)

audio_encoder = AudioEncoder()
audio_decoder = AudioDecoder()

encoder.train()
decoder.train()

to_pil = torchvision.transforms.ToPILImage()

loss = torch.nn.MSELoss()
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)

# Helper function that counts how many parameters a model has (pytorch does not natively support this)
def num_params(model):

    num = 0

    for i in model.parameters():
        num += 1
    
    return num


# Trains the image autoencoder. 

# clean_up_limit determines how frequently opencv objects are destroyed and the cache is cleared. 


def video_train(clean_up_limit=100):

    for path in os.listdir(project_path + '/video'):

        dataset = AudioVisualDataset(project_path + '/video/' + path, streams=0)
        data = DataLoader(dataset)

        resize = torchvision.transforms.Resize((512, 512))

        for _ in range(epochs):

            running_loss = 0.0
            
            clean_index = 0

            # Standard training loop

            for video in data:

                optimizer.zero_grad()
                video = video.to(device)

                img_out = encoder(video)
                img_out = decoder(img_out)

                video = resize(video)

                curr_loss = loss(img_out, video)
                running_loss += curr_loss.item()

                curr_loss.backward()
                optimizer.step()

                # Info display

                # TODO: Create a queue to iterate through batches while they are shown
                
                # torchvision returns an image normalized to [-0.5, 0.5], which is lucky, because opencv can open images in the range [-0.5, 0.5]
                # opencv reads (H, W, C), so we have to transpose the (C, H, W) tensor.

                display = img_out[0].cpu().detach().numpy().transpose(2, 1, 0) + 0.5
                truth = video[0].cpu().detach().numpy().transpose(2, 1, 0)

                print(running_loss)
                print(curr_loss.detach())

                cv.imshow('woa', display)
                cv.imshow('truth', truth)

                cv.waitKey(1)

            if clean_index > clean_up_limit:
                cv.destroyAllWindows() 
                torch.cuda.empty_cache()

            clean_index += 1

            torch.save({
                'encoder' : encoder.state_dict(),
                'decoder' : decoder.state_dict(),
                'optimizer' : optimizer.state_dict() }, project_path + '/models/model.pt')


def audio_training(clean_up_limit=dataset.audio_info['sample_rate'] * 10):

    # helper variable that determines the name of the saved file

    name = 0

    for path in os.listdir(project_path + '/video'):

        dataset = AudioVisualDataset(project_path + '/video/' + path, streams=1)
        data = DataLoader(dataset)

        resize = torch.nn.AdaptiveAvgPool1d(dataset.a_v_ratio)

        for _ in range(epochs):

            saved_data = []

            clean_index = 0

            for audio in dataset:

                saved_data.append(audio)

                if clean_index > clean_up_limit:

                    saved_data = torch.cat(saved_data, dim=1).view(2, -1).detach().clone()

                    torchaudio.save(
                        project_path + '/generated_audio/' + str(name) + '.wav', 
                        saved_data,
                        sample_rate=int(dataset.audio_info['sample_rate']),
                        channels_first=True)
                    
                    torch.cuda.empty_cache()
                    saved_data = []
                    clean_index = 0
                    name += 1
                
                clean_index += 1

# video_train()

audio_training()

#test()
