import torch
import torchvision
import os

# Audio Visual Correlator Network


# We want to loss a sound frame against an image frame. 

# First we project the sound and image into a space where they share the same dimension. 
# This will be done through an autoencoder. 

# The network, given the sound, creates a test output in that space. Training is done
# over the MSE loss of test vectors and truth vectors. 

BANDWIDTH_LIMIT = 64

KERNELS = [

    [1, 1],
    [3, 2],
    [3, 1],
    [1, 1],
    [3, 1],
    [3, 1],
    [3, 1]

]

CHANNELS_ENC = [

    [3, 3],
    [3, 4],
    [4, 8],
    [8, 8],
    [8, 12],
    [12, 16],
    [16, 16]
    
]

CHANNELS_DEC = [[pair[1], pair[0]] for pair in CHANNELS_ENC]

LAYERS_ENC = [channels + kernels for channels, kernels in zip(CHANNELS_ENC, KERNELS)]
LAYERS_DEC = [channels + kernels for channels, kernels in zip(CHANNELS_DEC, KERNELS)]


class ImageEncoder(torch.nn.Module):


    def __init__(self, device=torch.device('cpu')):

        super(ImageEncoder, self).__init__()

        self.relu1 = torch.nn.ReLU().to(device)
        self.relu2 = torch.nn.ReLU().to(device)

        self.pool1 = torch.nn.AdaptiveAvgPool2d((512, 512)).to(device)
        self.pool2 = torch.nn.AdaptiveAvgPool2d((BANDWIDTH_LIMIT, BANDWIDTH_LIMIT)).to(device)

        self.conv1 = torch.nn.Conv2d(*LAYERS_ENC[0]).to(device)
        self.conv2 = torch.nn.Conv2d(*LAYERS_ENC[1]).to(device)
        self.conv3 = torch.nn.Conv2d(*LAYERS_ENC[2]).to(device)
        self.conv4 = torch.nn.Conv2d(*LAYERS_ENC[3]).to(device)
        self.conv5 = torch.nn.Conv2d(*LAYERS_ENC[4]).to(device)
        self.conv6 = torch.nn.Conv2d(*LAYERS_ENC[5]).to(device)
        self.conv7 = torch.nn.Conv2d(*LAYERS_ENC[6]).to(device)



    def forward(self, x):

        out = x

        out = self.conv1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        # out = self.relu1(out)
        out = self.conv3(out)
        # out = self.conv4(out)

        # out = self.relu2(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        out = self.pool2(out)


        return out



class ImageDecoder(torch.nn.Module):

    def __init__(self, device=torch.device('cpu'), image_size=(512, 512)):

        super(ImageDecoder, self).__init__()

        self.pool = torch.nn.AdaptiveAvgPool2d(image_size).to(device)

        self.deconv1 = torch.nn.ConvTranspose2d(*LAYERS_DEC[6]).to(device)
        self.deconv2 = torch.nn.ConvTranspose2d(*LAYERS_DEC[5]).to(device)
        self.deconv3 = torch.nn.ConvTranspose2d(*LAYERS_DEC[4]).to(device)
        self.deconv4 = torch.nn.ConvTranspose2d(*LAYERS_DEC[3]).to(device)
        self.deconv5 = torch.nn.ConvTranspose2d(*LAYERS_DEC[2]).to(device)
        self.deconv6 = torch.nn.ConvTranspose2d(*LAYERS_DEC[1]).to(device)
        self.deconv7 = torch.nn.ConvTranspose2d(*LAYERS_DEC[0]).to(device)




    def forward(self, x):

        out = x.view(-1, 8, BANDWIDTH_LIMIT, BANDWIDTH_LIMIT)

        # out = self.deconv1(out)
        # out = self.deconv2(out)
        # out = self.deconv3(out)
        # out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        out = self.deconv7(out)

        out = self.pool(out)

        return out



class AudioEncoder(torch.nn.Module):

    # bandwidth_limit determines the size of the last pooled connection. It should be the same size as the video encoder's. 
    def __init__(self, device=torch.device('cpu')):
        
        super(AudioEncoder, self).__init__()
        
        self.pool1 = torch.nn.AdaptiveAvgPool1d(10000)
        self.pool2 = torch.nn.AdaptiveAvgPool1d(BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)

        self.relu1 = torch.nn.ReLU().to(device)

        self.conv1 = torch.nn.Conv1d(2, 2, 1, 1).to(device)
        self.conv2 = torch.nn.Conv1d(2, 4, 3, 2).to(device)
        self.conv3 = torch.nn.Conv1d(4, 8, 3, 1).to(device)
        self.conv4 = torch.nn.Conv1d(8, 12, 3, 1).to(device)
        self.conv5 = torch.nn.Conv1d(12, 16, 3, 1).to(device)


    def forward(self, x):

        out = x

        out = self.conv1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu1(out)
        out = self.conv3(out)

        out = self.pool2(out)

        return out



class AudioDecoder(torch.nn.Module):

    
    def __init__(self, device=torch.device('cpu')):

        super(AudioDecoder, self).__init__()

        self.deconv1 = torch.nn.ConvTranspose1d(16, 12, 3, 1).to(device)
        self.deconv2 = torch.nn.ConvTranspose1d(12, 8, 3, 1).to(device)
        self.deconv3 = torch.nn.ConvTranspose1d(8, 4, 3, 1).to(device)
        self.deconv4 = torch.nn.ConvTranspose1d(4, 2, 3, 2).to(device)
        self.deconv5 = torch.nn.ConvTranspose1d(2, 2, 1, 1).to(device)

    def forward(self, x):

        out = x.view(-1, 8, BANDWIDTH_LIMIT * BANDWIDTH_LIMIT)

        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)

        return out 


