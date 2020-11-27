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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = ImageEncoder().to(device)
decoder = ImageDecoder().to(device)

audio_encoder = AudioEncoder()
audio_decoder = AudioDecoder()

to_pil = torchvision.transforms.ToPILImage()

loss = torch.nn.MSELoss()
lr = 1e-2
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)

model_path = os.getcwd() + '/models'


# dynamic, might wanna find a way to crawl through /video
video_path = os.getcwd() + '/video/video.mp4'

dataset = VideoDataset(video_path)
data = DataLoader(dataset, batch_size=4)

# find a way to plug these values in
resize = torchvision.transforms.Resize((512, 512))

encoder.train()

decoder.train()

num_encoder_params = 0

for i in encoder.parameters():
    num_encoder_params += 1

num_decoder_params = 0

for i in decoder.parameters():
    num_decoder_params += 1







def visualize_parameters(k):
    
    encoder_params = encoder.parameters()
    decoder_params = decoder.parameters() 

    img = np.zeros((3, (num_encoder_params  + num_decoder_params) * k, k * k))

    for i in range(num_encoder_params  // k):

        img[1 : 3, i * k : (i + 1) * k, : ] = np.average(encoder_params[i].detach().numpy())

    for i in range(num_decoder_params // k):

        img[0 : 2, i * k : (i + 1) * k, : ] = np.average(decoder_params[i].detach().numpy())

    return img



def train(clean_up_index=100):

    encoder = ImageEncoder()
    decdoer = ImageDecoder()

    for path in os.listdir('./video/'):

        dataset = VideoDataset(os.getcwd() + '/video/' + path)
        data = DataLoader(dataset)

        for epoch in range(4):

            running_loss = 0.0

            for video, audio in data:

                optimizer.zero_grad()

                video = video.to('cuda')

                img_out = encoder(video)

                img_out = decoder(img_out)

                truth = resize(video)

                curr_loss = loss(img_out, truth)

                running_loss += curr_loss.item()

                curr_loss.backward()

                optimizer.step()

                print(running_loss)
                print(curr_loss.detach())

                # print(img_out)
                
                # img = visualize_parameters(10)

                cv.imshow('woa', (img_out[0].cpu().detach().numpy().transpose(2, 1, 0) + 0.5))
                cv.imshow('truth', truth[0].cpu().detach().numpy().transpose(2, 1, 0))

                cv.waitKey(1)

            if i > clean_up_index:
                cv.destroyAllWindows() 

            torch.save({
                'encoder' : encoder.state_dict(),
                'decoder' : decoder.state_dict(),
                'optimizer' : optimizer.state_dict() },'./models/model.pt')





def test():

    for img, audio in data:

        cv.imshow('woa', img[0].permute(2, 1, 0).detach().numpy())
        cv.waitKey(1)



train()

#test()