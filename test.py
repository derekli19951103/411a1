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
def initialize_single(size,names_set,act_nickname):
    y=[]
    x=ones((1,1024))
    for name,files in names_set.items():
        if name in act_nickname:
            for i in files[:size]:
                if name == 'baldwin':
                    y.extend([1])
                else:
                    y.extend([-1])
                pic = imread('cropped/' + i).flatten() / 255.
                x=vstack((x,pic))
    y=array(y)
    x=np.delete(x,0,0)
    x=x.T
    theta=zeros((1025, 1))
    return x,y,theta