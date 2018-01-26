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

def determine(data, name):
    (np.asarray(data) > 0.).sum() / 1024.
    if name == 'baldwin':
        return (np.asarray(data) > 0.).sum() > (np.asarray(data) <= 0.).sum()
    else:
        return (np.asarray(data) > 0.).sum() < (np.asarray(data) <= 0.).sum()

files = os.listdir("cropped")
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
names_set = {}
for file in files:
    name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
    if name not in names_set:
        names_set[name] = [file]
    else:
        names_set[name].append(file)

def f(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return sum((y - dot(theta.T, x)) ** 2)


def df(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    # print "y ", y.shape
    # print "t ", theta.shape
    # print "x_b ", dot(theta.T, x).shape
    # print "x ", x.shape
    return -2 * sum((y - dot(theta.T, x)) * x, 1)


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = 100000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        # if iter % 500 == 0:
        #     print "Iter", iter
        #     print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

theta_face=hstack((zeros((33, 1)),vstack((zeros((1, 32)),imread("cropped/baldwin49.jpg")[:,:,0]/255.))))
for name,files in names_set.items():
    if name=='baldwin':
        y = array([[1] * 32] * 33)
    else:
        y = array([[-1] * 32] * 33)
    for i in files[:70]:
        pic1 = imread('cropped/' + i)
        pic1 = pic1[:, :, 0] / 255.
        x = pic1
        theta_face = grad_descent(f, df, x, y, theta_face, 0.0000010)

imsave("t4.png",theta_face)