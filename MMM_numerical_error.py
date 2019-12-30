
from sympy import *
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode

global dx
global tau
tau = 100
dx = 0.1
num = int(2 / dx) + 1

def ta(t):
    return -math.log(3.54466 - t)

def xi(x, t):
    return x / math.sqrt(4 * ta(t) * (3.54466 - t))

def u_exact(x, t):
    return ta(t) - math.log(1 + xi(x, t)**2)

def MMPDE(u, x, dt):
    num = int(u.shape[0])
    A = np.zeros((num, num))
    for i in range(num - 1):
        A[i][i + 1] = -dt * 0.5 * (math.exp(u[i + 1]) + math.exp(u[i])) / (math.exp(u[i]) * tau * dx**2)
        A[i + 1][i] = -dt * 0.5 * (math.exp(u[i]) + math.exp(u[i - 1])) / (math.exp(u[i]) * tau * dx**2)
    for i in range(1, num - 1):
        A[i][i] = 1 - A[i][i + 1] - A[i][i - 1]
    A[0][0] = 1
    A[num - 1][num - 1] = 1
    A = np.mat(A)
    x = np.mat(x)
    U = A.I * x.T
    U = np.array(U)
    soln = np.zeros(num)
    soln[0] = -1
    soln[num - 1] = 1
    for i in range(1, num - 1):
        soln[i] = U[i][0]
    return soln

def f(t, y):
    n = int(y.shape[0])
    U = np.zeros(n)
    U[0] = 0
    U[n - 1] = 0
    for i in range(1, n - 1):
        U[i] = (y[i + 1] - 2 * y[i] + y[i - 1]) / dx**2 + math.exp(y[i])
    return U

def amm_f(t, y, x):
    n = int(y.shape[0])
    U = np.zeros(n)
    U[0] = 0
    U[n - 1] = 0
    for i in range(1, n - 1):
        U[i] = ( 0.5 * (math.exp(y[i + 1]) + math.exp(y[i])) * (x[i + 1] - x[i]) - 0.5 * (math.exp(y[i]) + math.exp(y[i - 1])) * (x[i] - x[i - 1])) / (math.exp(y[i]) * tau * dx**2) * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]) + 2 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])) / (x[i + 1] - x[i - 1]) + math.exp(y[i])
    return U


if __name__ == "__main__":
    t_0 = 0
    e1 = 0
    e2 = 0
    X = np.linspace(-1, 1, num)
    mesh = X
    y_0 = np.zeros(num)
    y_1 = np.zeros(num)
    y_2 = np.zeros(num)
    r = ode(f).set_integrator('zvode', method='bdf')
    r.set_initial_value(y_1, t_0)
    #s = ode(f).set_integrator('zvode', method='bdf')
    #s.set_initial_value(y_2, t_0)
    t_1 = 3.54466
    dt = 0.0001
    while r.successful() and r.t <= t_1:
        r.integrate(r.t + dt)
        print(r.t)
        #s.integrate(s.t + dt)
        #mesh = MMPDE(s.y, mesh, dt)
        Y = r.y
        T = r.t
        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(Y, T).set_f_params(mesh)
        if abs(r.t - 3.5) < 1e-6:
            for i in range(1, num - 1):
                y_0[i] = u_exact(-1 + i * dx, r.t)
            for i in range(int((num - 1) / 2), int((num - 1) * 3 / 4)):
                e1 += (y_0[i] - r.y[i])**2
            e1 = e1 * 4 / int(num - 1)
            print(math.sqrt(e1))
            #for i in range(1, num - 1):
            #    y_0[i] = u_exact(mesh[i], s.t)
            #for i in range(int((num - 1) / 2), int((num - 1) * 3 / 4)):
            #    e2 += (y_0[i] - s.y[i])**2
            #e2 = e2 * 4 / int(num - 1)
            #print(math.sqrt(e2))
            break
