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

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
files=os.listdir("cropped")
training=0
validating=0
testing=0
actor=""
for file in files:
    try:
        pic = imread("cropped/" + file)
        name=''.join([i for i in file if not i.isdigit()])
        if actor=="":
            actor=name
            training+=1
            mpimg.imsave("training/" + file, pic)
        if actor==name:
            if training<100:
                training+=1
                mpimg.imsave("training/"+file,pic)
            if validating<10:
                validating+=1
                mpimg.imsave("validating/"+file,pic)
            if testing<10:
                testing+=1
                mpimg.imsave("testing/"+file,pic)
        if actor!=name:
            actor=name
            training=1
            mpimg.imsave("training/"+file,pic)
            validating=0
            testing=0
    except IOError:
        pass
