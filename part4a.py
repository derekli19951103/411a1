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

def reconstruct(theta):
    theta = theta[0]
    result = []
    interval = range(1, 1025, 32)
    interval.append(1025)
    for i in range(len(interval)):
        if i < len(interval) - 1:
            result.append(theta[interval[i]:interval[i + 1]])
        else:
            break
    return array(result)

files = os.listdir("cropped")
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
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

def f(x, y, theta):
    return sum((y - dot(theta, x.T)) ** 2)


def df(x, y, theta):
    return -2 * sum((y - dot(theta, x.T)) * x.T, 1)


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = 100000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        iter += 1
    return t

y=[]
x=ones((1,1024))
for name,files in names_set.items():
    for i in files[:70]:
        if name == 'baldwin':
            y.extend([1])
        else:
            y.extend([-1])
        pic = imread('cropped/' + i)[:, :, 0] / 255.
        x=vstack((x,pic.flatten()))
y=array(y)
x=np.delete(x,0,0)
x = np.c_[ones((len(training), 1)), x]
theta=zeros((1,1025))
theta = grad_descent(f, df, x, y, theta, 0.0000010)
lossHistory = f(x, y, theta)


imsave("t3.png",reconstruct(theta))