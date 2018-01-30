import os
import urllib
from pylab import *
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave

all_act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell',
           'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher',
           'America Ferrera']


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255.


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


testfile = urllib.URLopener()

if not os.path.exists("uncropped"):
    os.makedirs("uncropped")
if not os.path.exists("cropped"):
    os.makedirs("cropped")

for a in all_act:
    name = a.split()[1].lower()
    i = 0
    for line in open("faces_subset.txt"):
        if a in line:
            filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
            timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 30)
            if not os.path.isfile("uncropped/" + filename):
                continue
            try:
                x1, y1, x2, y2 = line.split()[-2].split(',')
                pic = imread('uncropped/' + filename)
                pic = pic[int(y1):int(y2), int(x1):int(x2)]
                try:
                    pic = rgb2gray(pic)
                    pic = imresize(pic, (32, 32))
                    imsave("cropped/" + filename, pic)
                    print filename
                except (ValueError, IndexError) as e:
                    pass
            except IOError:
                pass
            i += 1
print "part1 done downloading"

files = os.listdir("uncropped")
names_set_u = {}
for file in files:
    name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
    if name not in names_set_u:
        names_set_u[name] = [file]
    else:
        names_set_u[name].append(file)
print "uncropped:"
for name,files in names_set_u.items():
    print name,len(files)
files = os.listdir("cropped")
names_set_c = {}
for file in files:
    name = ''.join([i for i in file if not i.isdigit()]).split('.')[0]
    if name not in names_set_c:
        names_set_c[name] = [file]
    else:
        names_set_c[name].append(file)
print "cropped:"
for name,files in names_set_c.items():
    print name,len(files)
