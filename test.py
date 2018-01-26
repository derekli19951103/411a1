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
    interval = range(0, 1024, 32)
    interval.append(1024)
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


def f(x, y, theta):
    return sum((y - dot(theta, x.T)) ** 2)


def df(x, y, theta):
    return -2 * sum((y - dot(theta, x.T)) * x.T, 1)


lossHistory = []


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
        lossHistory.append(f(x, y, theta2))
    return t


lossHistory = []
baldwin = names_set["baldwin"][70:80]
carell = names_set["carell"][70:80]
y = array([[1] * 10 + [-1] * 10])
pic1 = imread('cropped/' + baldwin[0])[:, :, 0] / 255.
x = pic1.flatten()
for i in range(1, 10):
    pic1 = imread('cropped/' + baldwin[i])[:, :, 0] / 255.
    x = vstack((x, pic1.flatten()))
for i in range(10):
    pic2 = imread('cropped/' + carell[i])[:, :, 0] / 255.
    x = vstack((x, pic2.flatten()))
x = np.c_[ones((20, 1)), x]
theta2 = zeros((1, 1025))
theta2 = grad_descent(f, df, x, y, theta2, 0.0000010)

baldwin = names_set["baldwin"][70:80]
carell = names_set["carell"][70:80]
result = 0
for i in range(10):
    pic1 = imread('cropped/' + baldwin[i])
    pic1 = pic1[:, :, 0] / 255.
    result1 = dot(theta2, np.insert(pic1, 0, 1))
    pic2 = imread('cropped/' + carell[i])
    pic2 = pic2[:, :, 0] / 255.
    result2 = dot(theta2, np.insert(pic2, 0, 1))
    if result1 > 0:
        result += 1
    if result2 < 0:
        result += 1

print "===============Validating Set================"
print "ACCURARY for Baldwin: Baldwin:", result / 20.
print "Loss: ", lossHistory[-1], "\n"
imsave("test.png", reconstruct(theta2))