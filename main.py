import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv 
import os

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from PIL import Image

from AVDataset import *
from AVC import *

# static things, probably wont change between training loops


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = ImageEncoder().to(device)
decoder = ImageDecoder().to(device)

audio_encoder = AudioEncoder()
audio_decoder = AudioDecoder()

encoder.train()
decoder.train()

to_pil = torchvision.transforms.ToPILImage()

loss = torch.nn.MSELoss()
lr = 1e-2
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)

project_path = os.getcwd()

model_path = project_path + '/models'

# find a way to plug these values in
resize = torchvision.transforms.Resize((512, 512))


num_encoder_params = 0
num_decoder_params = 0

for i in encoder.parameters():
    num_encoder_params += 1

num_decoder_params = 0

for i in decoder.parameters():
    num_decoder_params += 1


def visualize_parameters(n):
    
    for i, p in enumerate(encoder.parameters()):

        if i is n:

            param_values = p.data.detach().permute(2, 1, 0).cpu().numpy()

            param_values[: , 0 : 2] = np.add.reduce(param_values, 1)

            for param in param_values:

                cv2.imshow(str(param), param_values[i][0 : 2])

                cv2.waitKey()

    return img



def train(clean_up_index=100):

    for path in os.listdir(project_path + '/video'):

        dataset = VideoDataset(project_path + '/video' + path)
        data = DataLoader(dataset)

        for epoch in range(4):

            running_loss = 0.0
            
            clean_index = 0

            for video, audio in data:

                optimizer.zero_grad()
                video = video.to(device)

                img_out = encoder(video)
                img_out = decoder(img_out)

                truth = resize(video)

                curr_loss = loss(img_out, truth)
                running_loss += curr_loss.item()

                curr_loss.backward()
                optimizer.step()

                print(running_loss)
                print(curr_loss.detach())

                img = Image.fromarray(img_out[0].cpu().detach().numpy().transpose(2, 1, 0) * 255 + 127, 'RGB')
                img.show()

                # print(img_out)
                
                # img = visualize_parameters(8)

                # cv.imshow('woa', (img_out[0].cpu().detach().numpy().transpose(2, 1, 0) + 0.5))
                # cv.imshow('truth', truth[0].cpu().detach().numpy().transpose(2, 1, 0))

                # cv.waitKey(1)

            if clean_index > clean_up_index:
                cv.destroyAllWindows() 
                torch.cuda.empty_cache()

            clean_index += 1

            torch.save({
                'encoder' : encoder.state_dict(),
                'decoder' : decoder.state_dict(),
                'optimizer' : optimizer.state_dict() }, project_path + '/models/model.pt')



def test():

    for img, audio in data:

        cv.imshow('woa', img[0].permute(2, 1, 0).detach().numpy())
        cv.waitKey(1)


for p in encoder.parameters():
    print(p.data.shape)

train()

#test()