import torch 
import torchvision
import numpy as np
import torchaudio
import sys
import cv2 as cv 
import os
import argparse
import torch.multiprocessing as mp
import output

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from PIL import Image

from AVDataset import *
from AVC import *

args = argparse.ArgumentParser()

args.add_argument("input", help="Determines if audio or video is encoded. Takes values 'video' or 'audio'.")
args.add_argument("output", help="Determines if audio or video is encoded. Takes values 'video' or 'audio'.")

args.add_argument("--path", help="Trains/evals on each file in this folder, in alphabetical order. Defaults to pwd + /video/", default=os.getcwd() + "/video/", type=str)
args.add_argument("--device", help="Determines if the network will run on the GPU. Can be set to 'cuda', or 'cpu', defaults to 'cpu'.", default='cpu', type=str)

args.add_argument("--lr", help="Sets learning rate. Defaults to 1e-4.", default=1e-4, type=float)
args.add_argument("--batch-size", help="Sets batch size. Defaults to 4.", default=4, type=int)
args.add_argument("--epochs", help="Sets number of epochs. Defaults to 7.", default=7, type=int)
args.add_argument("--clean", help="Number of video/audio frames run before windows are recreated and cache is emptied. Defaults to 100.", default=100, type=int)

#TODO: Add audio output stream
args.add_argument("--display-truth", help="If set, will create an opencv window with truth data.", action='store_true')
args.add_argument("--display-out", help="If set, will create an opencv window with network output data.", action='store_true')
args.add_argument("--image-size", help="Define the height and width of the output image in pixels. Defaults to 512 x 512.", default=(512, 512), nargs="+")

args.add_argument("--eval", help="Sets the network to evaluate each file in path. You must have this or --train for the network to do anything.", action='store_true')

args = args.parse_args()


# def _display_video(name, frame):

#     cv.imshow(name, frame)

def display_video(_args, out, truth):
    
    if args.display_out:

        cv.imshow('out', out.cpu().detach().numpy().transpose(2, 1, 0))
        
        cv.waitKey(0)

    if args.display_out:

        cv.imshow('truth', truth.cpu().detach().numpy().transpose(2, 1, 0))

        cv.waitKey(0)


# Saves a model as a .pt file in the /models directory. Only saves the most recent one.
def save_model(_args):

    torch.save({
        'encoder' : _args['encoder'].state_dict(),
        'decoder' : _args['decoder'].state_dict(),
        'optimizer' : _args['optimizer'].state_dict() }, os.getcwd() + '/models/{_args[\'encoder\'] + _args[\'decoder\']}_model.pt')

def eval(_args, i):

    encoder_out = _args['encoder'](i)
    decoder_out = _args['decoder'](encoder_out)

    return decoder_out

# Trainer callback used in the data loop
# Inputs are _args, truth, an input tensor, and the cleaning index (optional)
def train(_args, i, t, c=-1):

    _args['optimizer'].zero_grad()    

    out = eval(_args, i)

    _args['current-loss'] = _args['loss'][args.output](out, t)

    _args['current-loss'].backward()
    _args['optimizer'].step()

    _args['running-loss'] += _args['current-loss'].item()
    total, curr = _args['running-loss'], _args['current-loss'].item()

    print(f'current loss:  {curr}     total loss: {total}     iter {c}'.format(curr, total, c))
        
    return out

# Converts the dataset into the DataLoader type, and loops over it
# Passes in _args and batched data to the callback

# The callback evaluates or trains a network.

def loop(_args, callback):

    for _ in range(args.epochs):

        for i, t in _args['data']:
            out = callback(_args, i, t)
            display_video(_args, out[0], t[0])



def parallel_train(_args, i, t, oq, c=-1):

    _args['optimizer'].zero_grad()    

    encoder_out = _args['encoder'](i.to(_args['device']))
    out = _args['decoder'](encoder_out)

    for img in out:
        oq.put(img.clone().cpu().detach())

    _args['current-loss'] = _args['loss'](out, t.to(_args['device']))

    _args['current-loss'].backward()
    _args['optimizer'].step()

    del i
    del t
    del out

    _args['running-loss'] += _args['current-loss'].item()
    total, curr = _args['running-loss'], _args['current-loss'].item()

    print(f'current loss:  {curr}     total loss: {total}     iter {c}'.format(curr, total, c))
        

def parallel_loop(_args, tq, oq, callback):

    for _ in range(args.epochs):

        for i, t in _args['data']:

            callback(_args, i, t, oq)

            for truth in t:
                tq.put(truth)
            


# Loads the ith data file in directory path.
# returns (DataLoader, info)
# where info is a 2 tuple containing 
# audio_info, visual_info

# which are documented in AVDataset.py
def load_data(_args, pathname):

    dataset = AudioVisualDataset(pathname, _args['input'] + _args['output'], args.image_size)

    _args['info'] = [dataset.audio_info, dataset.visual_info, dataset.a_v_ratio]
    _args['data'] = DataLoader(dataset, batch_size=args.batch_size)



def main():

    _args = {

        "encoder" : None,
        "decoder" : None,

        "device" : torch.device(args.device),

        "current-loss" : 0.0,
        "running-loss" : 0.0,

        "input" : args.input,
        "output" : args.output

    }

    # We need to load data to do some init things
    load_data(_args, args.path +  "/" + os.listdir(args.path)[0])

    _args['streams']       = args.input + args.output
    _args['display_out']   = args.display_out
    _args['display_truth'] = args.display_truth

    # Encoder setting
    if args.input == 'audio':

        _args['encoder'] = AudioEncoder()

    else:

        _args['encoder'] = ImageEncoder()


    # Decoder setting
    if args.output == 'video':

        _args['decoder'] = ImageDecoder(args.image_size)
    
    else:

        _args['decoder'] = AudioDecoder(_args['info'][2])


    # Device setting
    if args.device is not None:
        _args['device'] = torch.device(args.device)


    _args['encoder'].to(_args['device'])
    _args['decoder'].to(_args['device'])


    # Train/eval loop
    if args.eval:

        _args['encoder'].eval()
        _args['decoder'].eval()

    else:

        _args['encoder'].train()
        _args['decoder'].train()
        _args['optimizer'] = Adam(list(_args['encoder'].parameters()) + list(_args['decoder'].parameters()), args.lr)

        tq = mp.Queue(maxsize=100)
        oq = mp.Queue(maxsize=100)

        p_display = mp.Process(target=output.parallel_display, args=(_args, tq, oq))
        p_display.start()

        for name in os.listdir(args.path):
            
            load_data(_args, args.path + '/' + name)
        
            _args['loss'] = torch.nn.MSELoss()
            
            parallel_loop(_args, tq, oq, parallel_train)

        loop(_args, eval)

main()
