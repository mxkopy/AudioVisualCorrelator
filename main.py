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

video_encoder = ImageEncoder().to(device)
video_decoder = ImageDecoder().to(device)
audio_encoder = AudioEncoder().to(device)
audio_decoder = AudioDecoder().to(device)

video_encoder.train()
video_decoder.train()
audio_encoder.train()
audio_decoder.train()

to_pil = torchvision.transforms.ToPILImage()

loss = torch.nn.MSELoss()
video_optimizer = Adam(list(video_encoder.parameters()) + list(video_decoder.parameters()), lr)
audio_optimizer = Adam(list(audio_encoder.parameters()) + list(audio_decoder.parameters()), lr)


# Trains the image autoencoder. 

# clean_up_limit determines how frequently opencv objects are destroyed and the cache is cleared. 


def video_train(clean_up_limit=10000):

    for path in os.listdir(project_path + '/video'):

        dataset = AudioVisualDataset(project_path + '/video/' + path, streams=0)
        data = DataLoader(dataset)

        resize = torchvision.transforms.Resize((512, 512))

        for _ in range(epochs):

            running_loss = 0.0
            
            clean_index = 0

            # Standard training loop

            for video in data:

                video_optimizer.zero_grad()
                video = video.to(device) / 255

                img_out = video_encoder(video)
                img_out = video_decoder(img_out)

                video = resize(video)

                curr_loss = loss(img_out, video)
                running_loss += curr_loss.item()

                curr_loss.backward()
                video_optimizer.step()

                # Info display

                # TODO: Create a queue to iterate through batches while they are shown
                
                # torchvision returns an image normalized to [-0.5, 0.5], which is lucky, because opencv can open images in the range [-0.5, 0.5]
                # opencv reads (H, W, C), so we have to transpose the (C, H, W) tensor.

                display = img_out[0].cpu().detach().numpy().transpose(2, 1, 0)
                truth = video[0].cpu().detach().numpy().transpose(2, 1, 0)

                print(running_loss)
                print(curr_loss.detach())

                cv.imshow('woa', display)
                cv.imshow('truth', truth)

                cv.waitKey(1)

                print(clean_index)

                if clean_index > clean_up_limit:

                    clean_index = 0

                    torch.save({
                        'encoder' : video_encoder.state_dict(),
                        'decoder' : video_decoder.state_dict(),
                        'optimizer' : video_optimizer.state_dict() }, project_path + '/models/video_model.pt')

                    cv.destroyAllWindows() 
                    torch.cuda.empty_cache()
                
                clean_index += 1

def audio_training(clean_up_limit=100):

    # helper variable that determines the name of the saved file

    name = 0

    for path in os.listdir(project_path + '/video'):

        dataset = AudioVisualDataset(project_path + '/video/' + path, streams=1)
        data = DataLoader(dataset, batch_size=batch_size)

        resize = torch.nn.AdaptiveAvgPool1d(dataset[0].shape[1])

        for _ in range(epochs):

            saved_data = []
            clean_index = 0
            running_loss = 0.0

            for audio in data:
                
                audio_optimizer.zero_grad()

                aud_out = audio_encoder(audio.to(device))
                aud_out = resize(audio_decoder(aud_out))

                curr_loss = loss(resize(aud_out), audio)

                saved_data.append(resize(aud_out).view(2, -1))
                
                running_loss += curr_loss.item()

                curr_loss.backward()
                audio_optimizer.step()

                print(clean_index)

                if clean_index > clean_up_limit:

                    saved_data = torch.cat(saved_data, dim=1).view(2, -1).detach().clone()

                    torchaudio.save(
                        project_path + '/generated_audio/' + str(name) + '.wav', 
                        saved_data,
                        sample_rate=int(dataset.audio_info['sample_rate']),
                        channels_first=True)
                    
                    saved_data = []
                    clean_index = 0
                    name += 1

                    torch.save({
                        'encoder' : audio_encoder.state_dict(),
                        'decoder' : audio_decoder.state_dict(),
                        'optimizer' : audio_optimizer.state_dict() }, project_path + '/models/audio_model.pt')

                
                clean_index += 1


def load_frankenstein(audio_encoder_path=project_path + '/models/audio_model.pt', video_decoder_path=project_path + '/models/video_model.pt'):

    audio_state_dict, video_state_dict = torch.load(audio_encoder_path), torch.load(video_decoder_path)
    audio_state_dict, video_state_dict = audio_state_dict['encoder'], video_state_dict['decoder']

    audio_encoder.load_state_dict(audio_state_dict)
    video_decoder.load_frankenstein(video_state_dict)


def eval_frankenstein(path='0.mp4', frame_limit=500):

    dataset = AudioVisualDataset(project_path + '/video/' + path, streams=1)

    audio_encoder.eval()
    video_decoder.eval()

    audio_encoder.to(device)
    video_decoder.to(device)

    saved_data = []

    for audio in dataset:

        if curr_frame > frame_limit:

            curr_frame = 0
            torchvision.io.write_video(project_path + '/generated_video/' + path, torch.cat(saved_data), fps=dataset.visual_info['fps'])
            saved_data = []

        aud_out = audio_encoder(audio.to(device))
        aud_out = aud_out.view(-1, 8, BANDWIDTH_LIMIT, BANDWIDTH_LIMIT)

        img_out = video_decoder(aud_out)

        img_out = img_out[0].cpu().detach().numpy().transpose(2, 1, 0)
        cv.imshow('vvoa', img_out)
        cv.waitKey(1)
    

load_frankenstein()
eval_frankenstein()

#video_train()

audio_training()

# video_train()

#test()
