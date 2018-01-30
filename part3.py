from grouping import *

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
names_set, training, validating, testing = group()


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


def grad_descent_for_test(f, df, x, y, init_t, alpha, iter):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = iter
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t)
        iter += 1
    return t


def generate_a(start, iterations):
    a = []
    for i in range(iterations):
        a.append(start)
        start += start
    return a


"""test max iterations and alpha"""

maxiters = [1000, 10000, 20000, 30000]
alphas = [0.0001, 0.00001, 0.000001, 1e-6]
losses = []
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
t = zeros((1025, 1))
for i in range(4):
    t = grad_descent_for_test(f, df, x, y, t, alphas[i], maxiters[i])
    losses.append(f(x, y, t))
plt.plot([1000, 10000, 20000, 30000], losses)
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.xlabel("iter=1000alpha=1e-4,iter=10000alpha=1e-5,iter=20000alpha=1e-6,iter=30000alpha=1e-6")
plt.ylabel("cost")
plt.title("iteration&alpha vs. cost")
plt.savefig("testiteration.png")
plt.gca().clear()

"""test alpha"""

alphas = generate_a(1e-6,6)
accuracy = []
baldwin = names_set["baldwin"][:70]
carell = names_set["carell"][:70]
y = array([[1] * 70 + [-1] * 70])
x = imread('cropped/' + baldwin[0]).flatten() / 255.
for i in range(1, 70):
    pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
    x = vstack((x, pic1))
for i in range(70):
    pic2 = imread('cropped/' + carell[i]).flatten() / 255.
    x = vstack((x, pic2))
x = x.T
t = zeros((1025, 1))
for i in range(6):
    t = grad_descent_for_test(f, df, x, y, t, alphas[i], 30000)
    baldwin = names_set["baldwin"][70:80]
    carell = names_set["carell"][70:80]
    result_t = 0
    for i in range(10):
        pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
        pic1 = np.insert(pic1, 0, 1)
        pic1 = pic1.T
        result1 = dot(t.T, pic1)
        pic1 = imread('cropped/' + carell[i]).flatten() / 255.
        pic1 = np.insert(pic1, 0, 1)
        pic1 = pic1.T
        result2 = dot(t.T, pic1)
        if result1 > 0:
            result_t += 1
        if result2 < 0:
            result_t += 1
    accuracy.append(result_t/20.)
plt.plot(alphas, accuracy)
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.title("alpha vs. accuracy")
plt.savefig("testalpha.png")

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
lossHistory2 = f(x, y, theta1)

"""start testing"""
baldwin = names_set["baldwin"][:70]
carell = names_set["carell"][:70]
result_t = 0
for i in range(70):
    pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
    pic1 = np.insert(pic1, 0, 1)
    pic1 = pic1.T
    result1 = dot(theta1.T, pic1)
    pic1 = imread('cropped/' + carell[i]).flatten() / 255.
    pic1 = np.insert(pic1, 0, 1)
    pic1 = pic1.T
    result2 = dot(theta1.T, pic1)
    if result1 > 0:
        result_t += 1
    if result2 < 0:
        result_t += 1

baldwin = names_set["baldwin"][70:80]
carell = names_set["carell"][70:80]
result_v = 0
for i in range(10):
    pic1 = imread('cropped/' + baldwin[i]).flatten() / 255.
    pic1 = np.insert(pic1, 0, 1)
    pic1 = pic1.T
    result1 = dot(theta1.T, pic1)
    pic1 = imread('cropped/' + carell[i]).flatten() / 255.
    pic1 = np.insert(pic1, 0, 1)
    pic1 = pic1.T
    result2 = dot(theta1.T, pic1)
    if result1 > 0:
        result_v += 1
    if result2 < 0:
        result_v += 1

print "accuracy for training set: ", result_t / 140.
print "accuracy for validating set: ", result_v / 20.
print "Loss for training: ", lossHistory1
print "Loss for validating: ", lossHistory2
