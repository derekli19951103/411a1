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
from scipy.misc import imsave

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



def grad_descent_before(f, df, x, y, init_t, alpha):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = 10
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        iter += 1
    return t

def grad_descent_after(f, df, x, y, init_t, alpha):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = 50000
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
            pic = imread('cropped/' + i) / 255.
            x=vstack((x,pic.flatten()))
y=array(y)
x=np.delete(x,0,0)
x=x.T
theta_b=zeros((1025, 1))
theta_b = grad_descent_before(f, df, x, y, theta_b, 0.0000010)

y=[]
x=ones((1,1024))
for name,files in names_set.items():
    if name in act_nickname:
        for i in files[:70]:
            if name == 'baldwin':
                y.extend([1])
            else:
                y.extend([-1])
            pic = imread('cropped/' + i) / 255.
            x=vstack((x,pic.flatten()))
y=array(y)
x=np.delete(x,0,0)
x=x.T
theta_a=zeros((1025, 1))
theta_a= grad_descent_after(f, df, x, y, theta_a, 0.0000010)


imsave("part4bb.png",np.resize(theta_b[1:], (32, 32)))
imsave("part4ba.png",np.resize(theta_a[1:], (32, 32)))