import torch
import torchvision
import os

# Audio Visual Correlator Network


# We want to loss a sound frame against an image frame. 

# First we project the sound and image into a space where they share the same dimension. 
# This will be done through an autoencoder. 

# The network, given the sound, creates a test output in that space. Training is done
# over the MSE loss of test vectors and truth vectors. 

class ImageEncoder(torch.nn.Module):


    def __init__(self):

        super(ImageEncoder, self).__init__()

        # Input is a sizable convolution, hopefully removes noise
        self.conv1 = torch.nn.Conv2d(3, 4, 5, 2, 1).cuda()
        self.relu1 = torch.nn.ReLU().cuda()
        self.pool1 = torch.nn.AdaptiveAvgPool2d((256, 256)).cuda()

        self.conv2 = torch.nn.Conv2d(4, 8, 3, 2).cuda()
        self.relu2 = torch.nn.ReLU().cuda()
        self.pool2 = torch.nn.AdaptiveAvgPool2d((64, 64)).cuda()

        self.conv3 = torch.nn.Conv2d(8, 16, 3, 1).cuda()
        self.relu3 = torch.nn.ReLU().cuda()
        self.pool3 = torch.nn.AdaptiveAvgPool2d((8, 8)).cuda()

        self.out = torch.nn.ReLU().cuda()


    def forward(self, x):

        out = self.conv1(x)
        # out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        # out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        # out = self.relu3(out)
        out = self.pool3(out)

        return out



class ImageDecoder(torch.nn.Module):

    def __init__(self):

        super(ImageDecoder, self).__init__()

        self.deconv1 = torch.nn.ConvTranspose2d(16, 8, 8, 3).cuda()
        self.relu1 = torch.nn.ReLU().cuda()
        self.pool1 = torch.nn.AdaptiveAvgPool2d((8, 8)).cuda()


        self.deconv2 = torch.nn.ConvTranspose2d(8, 4, 5, 2).cuda()
        self.relu2 = torch.nn.ReLU().cuda()
        self.pool2 = torch.nn.AdaptiveAvgPool2d((64, 64)).cuda()

        self.deconv3 = torch.nn.ConvTranspose2d(4, 3, 3, 1).cuda()
        self.relu3 = torch.nn.ReLU().cuda()
        self.pool3 = torch.nn.AdaptiveAvgPool2d((512, 512)).cuda()



    def forward(self, x):

        out = self.deconv1(x)
        # out = self.relu1(out)
        out = self.pool1(out)

        out = self.deconv2(out)
        # out = self.relu2(out)
        out = self.pool2(out)

        out = self.deconv3(out)
        # out = self.relu3(out)
        out = self.pool3(out)

        return out


class AudioEncoder(torch.nn.Module):


    def __init__(self):
        
        super(AudioEncoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(2, 4, 5, 2).cuda()
        self.relu1 = torch.nn.ReLU().cuda()

        self.conv2 = torch.nn.Conv1d(4, 8, 3, 1).cuda()
        self.relu2 = torch.nn.ReLU().cuda()

        self.conv3 = torch.nn.Conv1d(8, 16, 3, 1).cuda()
        self.relu3 = torch.nn.ReLU().cuda()

        self.pool1 = torch.nn.AdaptiveAvgPool1d(64)


    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.pool(out)

        return out.view(-1, 16, 8, 8)



class AudioDecoder(torch.nn.Module):

    
    def __init__(self):

        super(AudioDecoder, self).__init__()

        self.deconv1 = torch.nn.ConvTranspose1d(16, 8, 1, 8).cuda()
        self.relu1 = torch.nn.ReLU().cuda()

        self.deconv2 = torch.nn.ConvTranspose1d(8, 4, 1, 3).cuda()
        self.relu2 = torch.nn.ReLU().cuda()

        self.deconv3 = torch.nn.ConvTranspose1d(4, 3, 1, 3).cuda()
        self.relu3 = torch.nn.ReLU().cuda()

