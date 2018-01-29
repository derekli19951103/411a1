from grouping import *

files = os.listdir("cropped")
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_nickname = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
act_labels = {'bracco': [1, 0, 0, 0, 0, 0], 'gilpin': [0, 1, 0, 0, 0, 0], 'harmon': [0, 0, 1, 0, 0, 0],
              'baldwin': [0, 0, 0, 1, 0, 0], 'hader': [0, 0, 0, 0, 1, 0], 'carell': [0, 0, 0, 0, 0, 1]}
names_set, training, validating, testing = group()


def multi_f(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return sum(sum((dot(theta.T, x) - y) ** 2))


def multi_df(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return 2 * dot(x, (dot(theta.T, x) - y).T)


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


"""get theta"""
y = []
x = ones((1, 1024))
for name, files in names_set.items():
    if name in act_nickname:
        for i in files[:70]:
            y.append(act_labels[name])
            pic = imread('cropped/' + i).flatten() / 255.
            x = vstack((x, pic))
y = array(y).T
x = np.delete(x, 0, 0)
x = x.T
theta = zeros((1025, 6))
theta = grad_descent(multi_f, multi_df, x, y, theta, 0.0000010)
"""start testing"""
result_t = 0
for name, files in names_set.items():
    if name in act_nickname:
        for i in files[:70]:
            pic1 = imread('cropped/' + i).flatten() / 255.
            pic1 = np.insert(pic1, 0, 1)
            pic1 = pic1.T
            result1 = dot(theta.T, pic1)
            if np.argmax(result1) == act_labels[name].index(1):
                result_t += 1
print 'accuracy for training_set: ', result_t / 420.
result_v = 0
for name, files in names_set.items():
    if name in act_nickname:
        for i in files[70:80]:
            pic1 = imread('cropped/' + i).flatten() / 255.
            pic1 = np.insert(pic1, 0, 1)
            pic1 = pic1.T
            result1 = dot(theta.T, pic1)
            if np.argmax(result1) == act_labels[name].index(1):
                result_v += 1
print 'accuracy for validation set: ', result_v / 60.

imsave('bracco.png', np.resize(theta[:, 0].T[1:], (32, 32)))
imsave('gilpin.png', np.resize(theta[:, 1].T[1:], (32, 32)))
imsave('harmon.png', np.resize(theta[:, 2].T[1:], (32, 32)))
imsave('baldwin.png', np.resize(theta[:, 3].T[1:], (32, 32)))
imsave('hader.png', np.resize(theta[:, 4].T[1:], (32, 32)))
imsave('carell.png', np.resize(theta[:, 5].T[1:], (32, 32)))
print "part8 saved"
