import torch 
import torchvision
import numpy as np
import torchaudio
import sys
import cv2 as cv 
import os
import argparse
import multiprocessing as mp

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from PIL import Image

from AVDataset import *
from AVC import *

args = argparse.ArgumentParser()

args.add_argument("encoder", help="Determines if audio or video is encoded. Takes values 'video' or 'audio'.")
args.add_argument("decoder", help="Determines if audio or video is encoded. Takes values 'video' or 'audio'.")

args.add_argument("--path", help="Trains/evals on each file in this folder, in alphabetical order. Defaults to pwd + /video/", default=os.getcwd() + "/video/", type=str)
args.add_argument("--device", help="Determines if the network will run on the GPU. Can be set to 'cuda', or 'cpu', defaults to 'cpu'.", default='cpu', type=str)

args.add_argument("--lr", help="Sets learning rate. Defaults to 1e-4.", default=1e-4, type=float)
args.add_argument("--batch-size", help="Sets batch size. Defaults to 4.", default=4, type=int)
args.add_argument("--epochs", help="Sets number of epochs. Defaults to 7.", default=7, type=int)
args.add_argument("--clean", help="Number of video/audio frames run before windows are recreated and cache is emptied. Defaults to 100.", default=500, type=int)

#TODO: Add audio output stream
args.add_argument("--display-truth", help="If set, will create an opencv window with truth data.", action='store_true', default=True)
args.add_argument("--display-out", help="If set, will create an opencv window with network output data.", action='store_true', default=True)
args.add_argument("--image-size", help="Define the height and width of the output image in pixels. Defaults to 512 x 512.", default=(512, 512), nargs="+")

args.add_argument("--eval", help="Sets the network to evaluate each file in path.", action='store_true')

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


# def write_audio(args, name, data, path=os.getcwd() + '/generated_audio/'):

#     torchaudio.save(
#         path + str(name) + '.wav', 
#         data,
#         sample_rate=int(dataset.audio_info['sample_rate']),
#         channels_first=True)


# Saves a model as a .pt file in the /models directory. Only saves the most recent one.
def save_model(_args):

    name = args.encoder + "_" + args.decoder + ".pt"

    torch.save({
        'encoder' : _args['encoder'].state_dict(),
        'decoder' : _args['decoder'].state_dict(),
        'optimizer' : _args['optimizer'].state_dict() }, 
        os.getcwd() + "/models/" + name
    )
    

# Trainer callback used in the data loop
# Inputs are _args, truth, an input tensor, and the cleaning index (optional)
def train(_args, i, t, c=-1):

    _args['optimizer'].zero_grad()    

    encoder_out = _args['encoder'](i)
    out = _args['decoder'](encoder_out)

    _args['current-loss'] = _args['loss'][args.decoder](out, t)

    _args['current-loss'].backward()
    _args['optimizer'].step()

    _args['running-loss'] += _args['current-loss'].item()
    total, curr = _args['running-loss'], _args['current-loss'].item()

    print(f'current loss:  {curr}     total loss: {total}     iter {c}'.format(curr, total, c))
        
    return out


# Evaluates a model. 
# If the decoder is video, the output will be displayed
# Otherwise it will be saved in generated_audio

def eval(_args, i):

    return _args['decoder'](_args['encoder'](i))


# Converts the dataset into the DataLoader type, and loops over it
# Passes in _args and batched data to the callback

# The callback evaluates or trains a network.

def loop(_args, callback):

    for _ in range(args.epochs):

        for i, t in _args['data']:
            out = callback(_args, i, t)
            display_video(_args, out[0], t[0])


# Parallelized versions of above
def parallel_display(_args, tq, oq):

    clean_index = 0

    while True:

        if clean_index > args.clean:

            cv.destroyAllWindows() 
            torch.cuda.empty_cache()
            clean_index = 0

        if args.display_truth:

            truth = tq.get().cpu().detach().numpy()#

            cv.imshow('truth', truth.transpose(2, 1, 0))

            cv.waitKey(1)

        if args.display_out:

            out = oq.get().cpu().detach().numpy()

            cv.imshow('out', out.transpose(2, 1, 0))

            cv.waitKey(1)
        
        clean_index += 1



def parallel_train(_args, i, t, oq, save=-1):

    _args['optimizer'].zero_grad()    

    encoder_out = _args['encoder'](i)
    out = _args['decoder'](encoder_out)

    for img in out:
        oq.put_nowait(img.detach())

    _args['current-loss'] = _args['loss'](out, t)

    _args['current-loss'].backward()
    _args['optimizer'].step()

    _args['running-loss'] += _args['current-loss'].item()
    total, curr = _args['running-loss'], _args['current-loss'].item()

    print(f'current loss:  {curr}     total loss: {total}     iter {save}'.format(curr, total, save))
        

def parallel_eval(_args, i, t, oq):

    oq.put(_args['decoder'](_args['encoder'](i)))


def parallel_loop(_args, callback):

    tq = mp.Queue()
    oq = mp.Queue()
    save_index = 0

    # out = mp.Process(target=parallel_train, args=(_args, dq, oq))
    display = mp.Process(target=parallel_display, args=(_args, tq, oq))
    display.start()

    for _ in range(args.epochs):

        for i, t in _args['data']:

            for truth in t:
                tq.put_nowait(truth)

            callback(_args, i, t, oq, save_index)

            if save_index > args.clean:

                save_index = 0
                save_model(_args)
            display.join()

            save_index += 1



# Loads the ith data file in directory path.
# returns (DataLoader, info)
# where info is a 2 tuple containing 
# audio_info, visual_info

# which are documented in AVDataset.py
def load_data(_args, pathname):

    dataset = AudioVisualDataset(pathname, _args['streams'], _args['device'], args.image_size)

    _args['info'] = [dataset.audio_info, dataset.visual_info, dataset.a_v_ratio]
    _args['data'] = DataLoader(dataset, batch_size=args.batch_size)



def main():

    _args = {

        "encoder" : None,
        "decoder" : None,

        "device" : torch.device(args.device),

        "current-loss" : 0.0,
        "running-loss" : 0.0,

        "streams" : 0
    }


    # We need to load data to do some init things
    load_data(_args, args.path +  "/" + os.listdir(args.path)[0])


    # Encoder setting
    if args.encoder == 'audio':

        _args['encoder'] = AudioEncoder()

    else:

        _args['encoder'] = ImageEncoder()
        _args['streams'] = 3


    # Decoder setting
    if args.decoder == 'video':

        _args['decoder'] = ImageDecoder(args.image_size)
        _args['streams'] = (_args['streams'] + 2) % 4
    
    else:

        _args['decoder'] = AudioDecoder(_args['info'][2])


    # Device setting
    if args.device is not None:
        _args['device'] = torch.device(args.device)

    
    # Train/eval loop
    if not args.eval:

        _args['encoder'].train()
        _args['decoder'].train()
        _args['optimizer'] = Adam(list(_args['encoder'].parameters()) + list(_args['decoder'].parameters()), args.lr)

        for name in os.listdir(args.path):
            
            load_data(_args, args.path + '/' + name)

            _args['loss'] = torch.nn.MSELoss()

            parallel_loop(_args, parallel_train)


    else:

        path = os.getcwd() + "/models/"

        encoders = {
            'video' : torch.load(path + "/video_video.pt")['encoder'],
            'audio' : torch.load(path + "/audio_audio.pt")['encoder']
        }

        decoders = {
            'video' : torch.load(path + "/video_video.pt")['decoder'],
            'audio' : torch.load(path + "/audio_audio.pt")['decoder']
        }

        encoders = {
            'video' : ImageEncoder().load_state_dict(encoders['video']),
            'audio' : AudioEncoder().load_state_dict(encoders['audio'])
        }

        decoders = {
            'video' : ImageDecoder(args.image_size).load_state_dict(encoders['video']),
            'audio' : AudioDecoder(_args['info'][2]).load_state_dict(encoders['audio'])
        }

        _args['encoder'], _args['decoder'] = encoders[args.encoder], decoders[args.decoder]

        _args['encoder'].eval()
        _args['decoder'].eval()

        parallel_loop(_args, parallel_eval)
    

main()
