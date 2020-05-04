
import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy
from keras.models import load_model

model = load_model('modelsize_model.h5')
# model - load_model('flop_model.h5)

print("cvxpy version:", cvxpy.__version__)
#with open('reg.pl',"rb") as file:
#    reg = pl.load(file)

l_bar = 2.0  # length of bar
M = 1.0  # [kg]
m = 0.3  # [kg]
g = 9.8  # [m/s^2]

Q = np.diag([0.0, 0.01, 0.01, 0.0])
R = np.diag([0.01])
nx = 4   # number of state
nu = 1   # number of input
T = 30  # Horizon length
delta_t = 0.1  # time tick

animation = True

Ypred = np.load('Ypred.npy')
X_test = np.load('Xtest.npy')

def main():
    x0 = np.array([
        [0.0],
        [0.0],
        [0.3],
        [0.0]
    ])

    x = np.copy(x0)

    for i in range(X_test.shape[0]):
        print(X_test[i,:])
        x = simulation(X_test[i,:])

        if animation:
            plt.clf()
            px = float(x[0])
            theta = float(x[2])
            show_cart(px, theta)
            plt.xlim([-10, 2.0])
            plt.pause(0.1)


def simulation(x):

    A, B = get_model_matrix()
    A = A.reshape([4,4])
    x = x.reshape([4,1])
    B = B.reshape([4,1])
    u = model.predict(x.reshape([1,4]))
    u = u.reshape([1,1])
    x = np.dot(A,x)+np.dot(B,u)
    print(x.shape)
    return x

def get_nparray_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()


def get_model_matrix():

    # Model Parameter
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B


def flatten(a):
    return np.array(a).flatten()


def show_cart(xt, theta):
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.1

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.matrix([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")

if __name__ == '__main__':
    main()