import sounddevice as sd
import numpy as np
from AVDataset import *

# rand = np.random.random((441000, 2))

AVD = AudioVisualDataset('/home/mxkopy/Programming/AudioVisualCorrelator/video/0.mp4', 'audiovideo', (512, 512))

for data in AVD:
    # print(data[0].dtype)

    sd.play(data[0].view(-1, 2).numpy().astype(np.int8), AVD.audio_info['sample_rate'])


sd.play(rand)