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
names_set = {}
for file in files:
    name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
    if name not in names_set:
        names_set[name] = [file]
    else:
        names_set[name].append(file)
baldwin = names_set["baldwin"][:70]
carell = names_set["carell"][:70]


def f(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return sum((y - dot(theta.T, x)) ** 2)


def df(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return -2 * sum((y - dot(theta.T, x)) * x, 1, keepdims=True)


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


"""get thetas"""
baldwin = names_set["baldwin"][:70]
carell = names_set["carell"][0:70]
y = array([[1] * 70 + [-1] * 70])
x = imread('cropped/' + baldwin[0]).flatten() / 255.
for i in range(1, 70):
    pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
    x = vstack((x, pic1))
for i in range(70):
    pic2 = imread('cropped/' + carell[i]).flatten() / 255.
    x = vstack((x, pic2))
x = x.T
theta1 = zeros((1025, 1))
theta1 = grad_descent(f, df, x, y, theta1, 1e-6)
lossHistory1 = f(x, y, theta1)

baldwin = names_set["baldwin"][70:80]
carell = names_set["carell"][70:80]
y = array([[1] * 10 + [-1] * 10])
x = imread('cropped/' + baldwin[0]).flatten() / 255.
for i in range(1, 10):
    pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
    x = vstack((x, pic1))
for i in range(10):
    pic2 = imread('cropped/' + carell[i]).flatten() / 255.
    x = vstack((x, pic2))
x = x.T
theta2 = zeros((1025, 1))
theta2 = grad_descent(f, df, x, y, theta2, 1e-6)
lossHistory2 = f(x, y, theta2)

"""start testing"""
baldwin = names_set["baldwin"][:80]
carell = names_set["carell"][:80]
result = 0
for i in range(80):
    pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
    pic1 = np.insert(pic1, 0, 1)
    pic1 = pic1.T
    result1 = dot(theta1.T, pic1)
    pic1 = imread('cropped/' + carell[i]).flatten() / 255.
    pic1 = np.insert(pic1, 0, 1)
    pic1 = pic1.T
    result2 = dot(theta1.T, pic1)
    if result1 > 0:
        result += 1
    if result2 < 0:
        result += 1

print "===============Result================"
print "accuracy: ", result / 160.
print "Loss for training: ", lossHistory1, "\n"
print "Loss for validating: ", lossHistory2, "\n"
imsave("training_set.png", np.resize(theta1[1:], (32, 32)))
