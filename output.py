import cv2 as cv
import numpy as np
import torch
import pyaudio

# Parallelized helper function for output display
# img_tensor is a torch tensor with shape [C, H, W]
def parallel_image_out(q, name='out'):

    img_tensor = q.get().cpu().detach().numpy()

    cv.imshow(name, img_tensor.transpose(2, 1, 0))
    cv.waitKey(1)


# audio_tensor is a torch tensor with shape [C, L]
# where L is the framerate
def parallel_audio_out(q, stream):

    sample = q.get().cpu().numpy().tobytes()
    stream.write(sample)


def init_streamers(_args, q, name="out"):

    streamer = None

    if _args['output'] == 'video':

        streamer = lambda : parallel_image_out(q, name=name)

    else:

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=_args['info'][0]['num_channels'], rate=_args['info'][0]['sample_rate'], output=True)
        streamer = lambda : parallel_audio_out(q, stream)

    return streamer

# Parallelized versions of above
def parallel_display(_args, tq, oq):

    tq_streamer, oq_streamer = None, None

    if _args['display_truth']:

        tq_streamer = init_streamers(_args, tq, name='truth')

    else:

        tq_streamer = lambda : tq.get()
    
    if _args['display_out']:

        oq_streamer = init_streamers(_args, oq, name='out')
    
    else:

        oq_streamer = lambda : oq.get()

    while True:

        tq_streamer()
        oq_streamer()