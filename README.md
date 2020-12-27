# AudioVisualCorrelator

A video decoder stacked on an audio encoder. Or an audio decoder stacked on a video encoder. Whatever floats your boat.

## Why

Have you ever wanted to know what sound looks like? No? Just me?

Now all of us can. Well, those who have linux.

# How 

Autoencoders essentially compress data. We can think of an encoder as 7zip - you put in a file, and a smaller version of that file comes out.

Now, what if you wrote your own version of 7zip that compresses it differently? You can still use the compressed file, but when you decompress it, you'll get something completely different.

That's what this is. Instead of 

movie -> movie.zip -> movie

We do 

movie -> movie.zip -> audio

We project an audio or video tensor to an intermediate representation space that is the same for both the audio and video encoder, so projection from that space back onto video or audio is arbitrary.

## Dependencies

You need Pytorch, torchvision and torchaudio for basic functionality. OpenCV is used for image display. 
FFMPEG is used to write audio.

You can find install instructions for each at:

https://github.com/pytorch/pytorch

https://github.com/pytorch/vision

https://github.com/pytorch/audio/

https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html

## Usage

```bash
python main.py input output
```

The input and output arguments can take on the values 'video' or 'audio'. By default, the network is a video autoencoder.
Adding --help will list more arguments.

By default, the network will train on the videos under AudioVisualCorrelator/video.