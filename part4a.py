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

files = os.listdir("cropped")
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_nickname=['bracco','gilpin','harmon','baldwin','hader','carell']
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
    x = vstack((ones((1, x.shape[1])), x))
    return sum((y - dot(theta.T, x)) ** 2)


def df(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return -2 * sum((y - dot(theta.T, x)) * x, 1).reshape((1025, 1))


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = 30000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        iter += 1
    return t

y=[]
x=ones((1,1024))
for name,files in names_set.items():
    if name in act_nickname:
        for i in files[:70]:
            if name == 'baldwin':
                y.extend([1])
            else:
                y.extend([-1])
            pic = imread('cropped/' + i).flatten() / 255.
            x=vstack((x,pic))
y=array(y)
x=np.delete(x,0,0)
x=x.T
theta_f=zeros((1025, 1))
theta_f = grad_descent(f, df, x, y, theta_f, 0.0000010)

y=[]
x=ones((1,1024))
for name,files in names_set.items():
    if name in act_nickname:
        for i in files[:2]:
            if name == 'baldwin':
                y.extend([1])
            else:
                y.extend([-1])
            pic = imread('cropped/' + i).flatten() / 255.
            x=vstack((x,pic))
y=array(y)
x=np.delete(x,0,0)
x=x.T
theta_2=zeros((1025, 1))
theta_2 = grad_descent(f, df, x, y, theta_2, 0.0000010)

imsave("full_set.png",np.resize(theta_f[1:], (32, 32)))
imsave("each2_set.png",np.resize(theta_2[1:], (32, 32)))