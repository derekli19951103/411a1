from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255.


if __name__ == "__main__":
    files = os.listdir("uncropped")
    for file in files:
        try:
            pic=imread("uncropped/"+file)
            if len(pic.shape)==3:
                pic = rgb2gray(pic)
                pic=imresize(pic,(32,32))
                mpimg.imsave("cropped/"+file,pic)
        except IOError:
            pass
