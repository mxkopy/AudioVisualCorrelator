# AudioVisualCorrelator

A video decoder stacked on an audio encoder. Or an audio decoder stacked on a video encoder. Whatever floats your boat.

## Dependencies

You need Pytorch, torchvision and torchaudio for basic functionality. OpenCV is used for image display. 
FFMPEG is used to write audio.

You can find install instructions for each at
https://github.com/pytorch/pytorch

https://github.com/pytorch/vision

https://github.com/pytorch/audio/

https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html

## Usage

```bash
python main.py encoder decoder --eval
```

The input and output arguments can take on the values 'video' or 'audio'. By default, the network is a video autoencoder.
Adding --help will list more arguments.

By default, the network will train on the videos under AudioVisualCorrelator/video.