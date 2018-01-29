from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


def group():
    files = os.listdir("cropped")
    names_set = {}
    for file in files:
        name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
        if name not in names_set:
            names_set[name] = [file]
        else:
            names_set[name].append(file)
    training = []
    validating = []
    testing = []
    count = 0
    for name, files in names_set.items():
        for f in files:
            if count < 70:
                training.append(f)
                count += 1
            if 70 <= count < 80:
                validating.append(f)
                count += 1
            if 80 <= count < 90:
                testing.append(f)
                count += 1
        count = 0
    return names_set, training, validating, testing
