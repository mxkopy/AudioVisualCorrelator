import torch 
import torchvision
import numpy as np
import torchaudio
import cv2 as cv 
import os
import argparse

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from PIL import Image

from AVDataset import *
from AVC import *

args = argparse.ArgumentParser()

args.add_argument("encoder", help="Determines if audio or video is encoded. Takes values 'video' or 'audio'.")
args.add_argument("decoder", help="Determines if audio or video is encoded. Takes values 'video' or 'audio'.")

args.add_argument("--path", help="Trains/evals on each file in this folder, in alphabetical order. Defaults to pwd + /video/", default=os.getcwd() + "/video/", type=str)
args.add_argument("--device", help="Determines if the network will run on the GPU. Can be set to 'cuda', or 'cpu', defaults to 'cpu'.", default=torch.device('cpu'),type=str)

args.add_argument("--lr", help="Sets learning rate. Defaults to 1e-4.", type=float)
args.add_argument("--batch-size", help="Sets batch size. Defaults to 4.", type=int)
args.add_argument("--epochs", help="Sets number of epochs. Defaults to 7.", type=int)
args.add_argument("--clean", help="Number of video/audio frames run before windows are recreated and cache is emptied. Defaults to 100.", type=int)

#TODO: Add audio output stream
args.add_argument("--display-truth", help="If set, will create an opencv window with truth data.", action='store_true')
args.add_argument("--display-out", help="If set, will create an opencv window with network output data.", action='store_true')
args.add_argument("--image-size", help="Define the height and width of the output image in pixels.", default=(512, 512), nargs="+")

args.add_argument("--train", help="Sets the network to train on each file in path. You must have this or --eval for the network to do anything.", action='store_true')
args.add_argument("--eval", help="Sets the network to evaluate each file in path. You must have this or --train for the network to do anything.", action='store_true')

args = args.parse_args()


def display_video(name, frame):

    cv.imshow(name, data)
    cv.waitKey(1)


def write_audio(args, name, data, path=os.getcwd() + '/generated_audio/'):

    torchaudio.save(
        path + str(name) + '.wav', 
        data,
        sample_rate=int(_args['t.audio_info['sample_rate']),
        channels_first=True)

def save_model(_args):

    torch.save({
        'encoder' : _args['encoder'].state_dict(),
        'decoder' : _args['decoder'].state_dict(),
        'optimizer' : _args['optimizer'].state_dict() }, os.getcwd() + '/models/video_model.pt')



def eval(_args, i):

    encoder_out = _args['encoder'](i)
    decoder_out = _args['decoder'](encoder_out)

    return decoder_out

# Trainer callback used in the data loop
# Inputs are _args, truth, an input tensor, and the cleaning index (optional)
def train(_args, i, t, c=-1):

    _args['optimizer'].zero_grad()    

    out = eval(_args, i)

    _args['current-loss'] = _args['loss'][args.decoder](out, t)
    _args['current-loss'].backward()

    _args['running-loss'] += _args['current-loss'].item()
    total, curr = _args['running-loss'], _args['current-loss'].item()

    _args['optimizer'].step()

    print(f'current loss:  {curr}     total loss: {total}     iter {i}'.format(curr, total, i))

    return out


# Converts the dataset into the DataLoader type, and loops over it
# Passes in _args and batched data to the callback

# The callback evaluates or trains a network.

def loop_over_data(_args, callback):

    for _ in range(_args['epochs']):

        for i, t in _args['data']:
            
            out = callback(_args, i, t)



# Loads the ith file in directory path.
# returns (DataLoader, info)
# where info is a 2 tuple containing 
# audio_info, visual_info

# which are documented in AVDataset.py
def load_data(_args, pathname):

    dataset = AudioVisualDataset(pathname, streams=_args['streams'] ,device=_args['device'])

    _args['info'] = [dataset.audio_info, dataset.visual_info, dataset.a_v_ratio]
    _args['data'] = DataLoader(dataset, batch_size=_args['batch-size'])



def main():

    _args = {

        "encoder" : None,
        "decoder" : None,

        "path" : os.getcwd() + '/video/',
        "device" : torch.device('cpu'),

        "lr" : 1e-4,
        "batch-size" : 4,
        "epochs" : 7,
        "clean" : 100,

        "display-truth" : False,
        "display-out" : False,
        "image-size" : (512, 512),

        "current-loss" : 0.0,
        "running-loss" : 0.0,

        "streams" : 0
    }

    # Dataset path
    if args.path is not None:

        _args['path'] = args.path



    # Network wont run if you don't tell it to
    if not args.train or args.eval:

        print("You must specify --train or --eval.")
        return


    # Encoder setting
    if args.encoder == 'audio':

        _args['encoder'] = AudioEncoder()

    else:

        _args['encoder'] = ImageEncoder()
        _args['streams'] = 3


    # Decoder setting
    if args.decoder == 'video':

        _args['decoder'] = ImageDecoder(image_size=args.image_size)
        _args['streams'] = (_args['streams'] + 2) % 4
    
    else:

        _args['decoder'] = AudioDecoder()


    # Device setting
    if args.clean is not None:

        _args['device'] = torch.device(args.device)


    # LR setting
    if args.lr is not None:
        _args['lr'] = args.lr
    

    # Batch size setting
    if args.batch_size is not None:
        _args['batch-size'] = args.batch_size


    # Epoch num setting
    if args.batch_size is not None:
        _args['epochs'] = args.epochs


    # Clean limit
    if args.clean is not None:

        _args['clean'] = args.clean


    # Display setting
    if args.display_truth:

        _args['display-truth'] = True

    if args.display_out:

        _args['display-out'] = True

    
    # Train/eval loop
    if args.train:

        _args['encoder'].train()
        _args['decoder'].train()
        _args['optimizer'] = Adam(list(_args['encoder'].parameters()) + list(_args['decoder'].parameters()), _args['lr'])
        loss = torch.nn.MSELoss()

        for name in os.listdir(_args['path']):
            
            load_data(_args, _args['path'] + '/' + name)
        
            # Resize operation is different depending on the dataset

            resize = [
                torch.nn.AdaptiveAvgPool1d(_args['info'][2]),
                torchvision.transforms.Resize((args.image_size[0], args.image_size[1]))
            ]

            _args['loss'] = {
                'audio': (lambda out, truth: loss(resize[0](out), truth)),
                'video': (lambda out, truth: loss(out, resize[1](truth)))
            }

            loop_over_data(_args, train)


    if args.eval:

        _args['encoder'].eval()
        _args['decoder'].eval()

        loop_over_data(_args, eval)
    

    _args['encoder'].to(_args['device'])
    _args['decoder'].to(_args['device'])


main()

# print(os.listdir('/home/mxkopy/Programming/AudioVisualCorrelator/video'))