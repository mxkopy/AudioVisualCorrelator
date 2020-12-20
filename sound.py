import pyaudio
import wave
import sys
import numpy as np
from AVDataset import *


CHUNK = 1024

wf = wave.open('/home/mxkopy/Programming/AudioVisualCorrelator/video/0.mp4.wav', 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=pyaudio.paFloat32,
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data
# data = wf.readframes(CHUNK)

AVD = AudioVisualDataset('/home/mxkopy/Programming/AudioVisualCorrelator/video/0.mp4', 'audiovideo', (512, 512))

for data in AVD:
    # print(data[0].dtype)
    sample = data[0].numpy().tobytes()#.astype(np.float32).tobytes()
    stream.write(sample)


# # play stream (3)
# while len(data) > 0:
#     stream.write(data)
#     data = wf.readframes(CHUNK)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()

