from grouping import *

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
all_act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell',
           'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher',
           'America Ferrera']
male = ['baldwin', 'hader', 'carell', 'radcliffe', 'butler', 'vartan']
female = ['bracco', 'gilpin', 'harmon', 'chenoweth', 'drescher', 'ferrera']
act_nickname = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
all_act_nickname = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell', 'radcliffe', 'butler', 'vartan',
                    'chenoweth', 'drescher', 'ferrera']
names_set, training, validating, testing = group()


def run_set(size):
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

    y = []
    x = ones((1, 1024))
    for name, files in names_set.items():
        if name in act_nickname:
            for i in files[:size]:
                if name in male:
                    y.extend([1])
                if name in female:
                    y.extend([-1])
                pic = imread('cropped/' + i) / 255.
                x = vstack((x, pic.flatten()))
    y = array(y)
    x = np.delete(x, 0, 0)
    x = x.T
    theta = zeros((1025, 1))
    theta = grad_descent(f, df, x, y, theta, 0.0000010)

    """start testing"""
    result_t = 0
    for name, files in names_set.items():
        if name in act_nickname:
            for i in files[:70]:
                pic1 = imread('cropped/' + i).flatten() / 255.
                pic1 = np.insert(pic1, 0, 1)
                pic1 = pic1.T
                result1 = dot(theta.T, pic1)
                if name in male and result1 > 0:
                    result_t += 1
                if name in female and result1 < 0:
                    result_t += 1
    result_v = 0
    for name, files in names_set.items():
        if name in act_nickname:
            for i in files[70:80]:
                pic1 = imread('cropped/' + i).flatten() / 255.
                pic1 = np.insert(pic1, 0, 1)
                pic1 = pic1.T
                result1 = dot(theta.T, pic1)
                if name in male and result1 > 0:
                    result_v += 1
                if name in female and result1 < 0:
                    result_v += 1
    return result_t / 420., result_v / 60., theta


"""run for various sizes"""
performances_t = []
performances_v = []
thetas = []
for i in range(2, 71, 2):
    p_t, p_v, t = run_set(i)
    performances_t.append(p_t)
    performances_v.append(p_v)
    thetas.append(t)
other = 0
for name, files in names_set.items():
    if name not in act_nickname and name in all_act_nickname:
        for i in files[:70]:
            pic1 = imread('cropped/' + i).flatten() / 255.
            pic1 = np.insert(pic1, 0, 1)
            pic1 = pic1.T
            result1 = dot(t.T, pic1)
            if name in male and result1 > 0:
                other += 1
            if name in female and result1 < 0:
                other += 1
print "performance on this 6 actors:", performances_t[-1]
print "performance on other 6 actors: ", other / 420.
plt.plot(range(2, 71, 2), performances_t)
plt.plot(range(2, 71, 2), performances_v)
plt.xlabel('training size per actor')
plt.ylabel('accuracy rate')
plt.title("training size per actor vs. accuracy rate of training and validating set")
plt.legend(['test on training set', 'test on validating set'], loc='upper left')
plt.savefig("part5.png")
print "part5 saved"
