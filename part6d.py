from grouping import *

names_set,training,validating,testing=group()
act_labels = {'bracco': [1, 0, 0, 0, 0, 0], 'gilpin': [0, 1, 0, 0, 0, 0], 'harmon': [0, 0, 1, 0, 0, 0],
              'baldwin': [0, 0, 0, 1, 0, 0], 'hader': [0, 0, 0, 0, 1, 0], 'carell': [0, 0, 0, 0, 0, 1]}
def multi_f(x, y, theta):
    return sum(sum((dot(theta.T, x) - y) ** 2))


def multi_df(x, y, theta):
    return 2 * dot(x, (dot(theta.T, x) - y).T)

def generate_h(start,iterations):
    hs=[]
    for i in range(iterations):
        hs.append(start)
        start /= 10.
    return hs


def finite_difference(x, y, theta, h):
    origin_t = multi_f(x, y, theta)
    theta = theta + np.full((theta.shape[0], theta.shape[1]), h)
    after_t = multi_f(x, y, theta)
    finite_diff = (after_t - origin_t)/h
    total_error = sum(finite_diff - multi_f(x, y, theta))
    return abs(total_error)/(1025*6*1.0)
pic1 = imread('cropped/' + names_set['bracco'][0]).flatten() / 255.
pic1 = np.insert(pic1, 0, 1)
pic1 = pic1.T
y=array(act_labels['bracco']).T
theta=zeros((1025,6))
hs=generate_h(0.1,10)
errors=[]
for h in hs:
    errors.append(finite_difference(pic1,y,theta,h))
plt.plot(range(1,11),errors)
plt.xlabel("h value in 10**(-xaxis)")
plt.ylabel("avg error")
plt.savefig("part7.png")
print "part6d saved"



