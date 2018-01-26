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
male=['baldwin','carell','hader']
female=['gilpin','harmon','bracco']
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
        if name in male:
            y.extend([1])
        else:
            y.extend([-1])
        pic = imread('cropped/' + i)[:, :, 0] / 255.
        x=vstack((x,pic.flatten()))
y=array(y)
x=np.delete(x,0,0)
x = np.c_[ones((len(training), 1)), x]
theta1=zeros((1,1025))
theta1 = grad_descent(f, df, x, y, theta1, 0.0000010)
lossHistory1 = f(x, y, theta1)


y=[]
x=ones((1,1024))
for name,files in names_set.items():
    for i in files[:70]:
        if name in male:
            y.extend([1])
        else:
            y.extend([-1])
        pic = imread('cropped/' + i)[:, :, 0] / 255.
        x=vstack((x,pic.flatten()))
y=array(y)
x=np.delete(x,0,0)
x = np.c_[ones((len(training), 1)), x]
theta2=zeros((1,1025))
theta2 = grad_descent(f, df, x, y, theta2, 0.0000010)
lossHistory2 = f(x, y, theta2)


"""start testing"""
result = 0
for name,files in names_set.items():
    for i in files[:70]:
        pic = imread('cropped/' + i)[:, :, 0] / 255.
        result1=dot(theta1,np.insert(pic.flatten(),0,1))
        if result1>0 and name in male:
            result+=1
        if result1<0 and name in female:
            result+=1

print "===============Training Set================"
print "ACCURARY :", result / len(training)
print "Loss: ", lossHistory1, "\n"

result = 0
for name,files in names_set.items():
    for i in files[70:80]:
        pic = imread('cropped/' + i)[:, :, 0] / 255.
        result2=dot(theta1,np.insert(pic.flatten(),0,1))
        if result2>0 and name in male:
            result+=1
        if result2<0 and name in female:
            result+=1

print "===============Validating Set================"
print "ACCURARY :", result / len(validating)
print "Loss: ", lossHistory2, "\n"