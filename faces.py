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
    if name == 'carell':
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
baldwin = names_set["baldwin"][:70]
carell = names_set["carell"][:70]


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


theta1 = np.random.random_sample((33, 33))
for i in range(70):
    pic1 = imread('cropped/' + baldwin[i])
    pic1 = pic1[:, :, 0] / 255.
    x = pic1
    y = array([[1] * 32] * 33)
    theta1 = grad_descent(f, df, x, y, theta1, 0.0000010)
    pic1 = imread('cropped/' + carell[i])
    pic1 = pic1[:, :, 0] / 255.
    x = pic1
    y = array([[-1] * 32] * 33)
    theta1 = grad_descent(f, df, x, y, theta1, 0.0000010)

baldwin = names_set["baldwin"][70:80]
carell = names_set["carell"][70:80]
theta2 = np.random.random_sample((33, 33))
for i in range(10):
    pic1 = imread('cropped/' + baldwin[i])
    pic1 = pic1[:, :, 0] / 255.
    x = pic1
    y = array([[1] * 32] * 33)
    theta2 = grad_descent(f, df, x, y, theta2, 0.0000010)
    pic1 = imread('cropped/' + carell[i])
    pic1 = pic1[:, :, 0] / 255.
    x = pic1
    y = array([[-1] * 32] * 33)
    theta2 = grad_descent(f, df, x, y, theta2, 0.0000010)

baldwin = names_set["baldwin"][:70]
carell = names_set["carell"][:70]
result = 0
for i in range(70):
    pic1 = imread('cropped/' + baldwin[i])
    pic1 = pic1[:, :, 0] / 255.
    result1 = dot(theta2.T, vstack((ones((1, pic1.shape[1])), pic1)))
    pic2 = imread('cropped/' + carell[i])
    pic2 = pic2[:, :, 0] / 255.
    result2 = dot(theta2.T, vstack((ones((1, pic2.shape[1])), pic2)))
    if determine(result1, 'baldwin'):
        result += 1
    if determine(result2, 'carell'):
        result += 1

print "===============Training Set================"
print "ACCURARY for Baldwin: Baldwin:", result / 140.

baldwin = names_set["baldwin"][70:80]
carell = names_set["carell"][70:80]
result = 0
for i in range(10):
    pic1 = imread('cropped/' + baldwin[i])
    pic1 = pic1[:, :, 0] / 255.
    result1 = dot(theta2.T, vstack((ones((1, pic1.shape[1])), pic1)))
    pic2 = imread('cropped/' + carell[i])
    pic2 = pic2[:, :, 0] / 255.
    result2 = dot(theta2.T, vstack((ones((1, pic2.shape[1])), pic2)))
    if determine(result1, 'baldwin'):
        result += 1
    if determine(result2, 'carell'):
        result += 1

print "===============Validating Set================"
print "ACCURARY for Baldwin: Baldwin:", result / 140.
imsave("t1.png", theta1)
imsave("t2.png", theta2)
