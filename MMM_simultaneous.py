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
tau = 1
dx = 0.01
num = int(2 / dx) + 1

def MMPDE(u, x):
    num = int(u.shape[0])
    A = np.zeros((num, num))
    for i in range(num - 1):
        A[i][i + 1] = - 0.5 * (math.exp(u[i + 1]) + math.exp(u[i]))
        A[i + 1][i] = - 0.5 * (math.exp(u[i + 1]) + math.exp(u[i]))
    for i in range(1, num - 1):
        A[i][i] = - A[i][i + 1] - A[i][i - 1]
    A[0][0] = 1
    A[num - 1][num - 1] = 1
    B = np.zeros((num, num))
    for i in range(num - 1):
        B[i][i + 1] = 0.5 * (math.exp(u[i + 1]) + math.exp(u[i])) * tau
        B[i + 1][i] = 0.5 * (math.exp(u[i + 1]) + math.exp(u[i])) * tau
    for i in range(1, num - 1):
        B[i][i] = - B[i][i + 1] - B[i][i - 1]
    B[0][0] = 1
    B[num - 1][num - 1] = 1
    A = np.mat(A)
    B = np.mat(B)
    C = B.I * A
    x = np.mat(x)
    U = C * x.T
    U = np.array(U)
    soln = np.zeros(num)
    soln[0] = 0
    soln[num - 1] = 0
    for i in range(1, num - 1):
        soln[i] = U[i][0]
    return soln

def f_mm(t, y):
    n = int(y.shape[0])
    U = np.zeros(n)
    num = int(n / 2)
    X = np.zeros(num)
    Y = np.zeros(num)
    for i in range(num):
        X[i] = y[i + num]
    for i in range(num):
        Y[i] = y[i]
    for i in range(num):
        U[i] = amm_f(t, Y, X)[i]
    for i in range(num, n):
        U[i] = MMPDE(Y, X)[i - num]
    return U

def f_mesh(t, y, u):
    n = int(y.shape[0])
    U = np.zeros(n)
    U[0] = 0
    U[n - 1] = 0
    for i in range(1, n - 1):
        U[i] = ( 0.5 * (math.exp(u[i + 1]) + math.exp(u[i])) * (y[i + 1] - y[i]) - 0.5 * (math.exp(u[i]) + math.exp(u[i - 1])) * (y[i] - y[i - 1])) / (math.exp(u[i]) * tau * dx**2)

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
        U[i] = MMPDE(y, x)[i] * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]) + 2 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])) / (x[i + 1] - x[i - 1]) + math.exp(y[i])
    return U


if __name__ == "__main__":
    X_1 = []
    X_2 = []
    X_3 = []
    X_4 = []
    X_5 = []
    U_max = []

    #amm
    y_0 = np.zeros(num)
    t_0 = 0
    t_1 = 3.5446
    dt = 0.0001
    X = np.linspace(-1, 1, num)
    mesh = X
    Z = np.zeros(2 * num)
    for i in range(num):
        Z[i] = y_0[i]
        Z[i + num] = X[i]
    r = ode(f_mm).set_integrator('zvode', method='bdf')
    r.set_initial_value(Z, t_0)

    plt.figure(figsize = (18, 4))
    plt.subplot(121)
    while r.successful() and r.t <= 0.03:
        r.integrate(r.t + dt)
        for i in range(num):
            mesh[i] = r.y[i + num]
        print(r.t)
        soln_t = np.zeros(num)
        for i in range(num):
            soln_t[i] = [i]
        if abs(r.t - 3) < 1e-6:
            plt.plot(X, soln_t, "g-", linewidth=1.0, label="T = 3")
        elif abs(r.t - 3.5) < 1e-6:
            plt.plot(X, soln_t, "r-", linewidth=1.0, label="T = 3.5")
        elif abs(r.t - 3.54) < 1e-6:
            plt.plot(X, soln_t, "b-", linewidth=1.0, label="T = 3.54")
        elif abs(r.t - 3.544) < 1e-6:
            plt.plot(X, soln_t, "y-", linewidth=1.0, label="T = 3.544")
        elif abs(r.t - 3.5446) < 1e-6:
            plt.plot(X, soln_t, "p-", linewidth=1.0, label="T = 3.5446")
        elif abs(r.t - 0.025) < 1e-6:
            plt.plot(X, soln_t, "p-", linewidth=1.0, label="T = 3.5446")
        x = mesh[int((num - 1) / 2)]
        x = math.log(abs(x))
        X_1.append(x)
        x = mesh[int((num - 1) / 2) + 1]
        x = math.log(abs(x))
        X_2.append(x)
        x = mesh[int((num - 1) / 2) + 2]
        x = math.log(abs(x))
        X_3.append(x)
        x = mesh[int((num - 1) / 2) + 3]
        x = math.log(abs(x))
        X_4.append(x)
        x = mesh[int((num - 1) / 2) + 4]
        x = math.log(abs(x))
        X_5.append(x)
        u = math.log(r.y[int((num - 1) / 2)])
        U_max.append(u)
    plt.legend()
    plt.title("The blow-up with MM by CXF")
    plt.subplot(122)
    plt.plot(U_max, X_1, "g-", linewidth=1.0)
    plt.plot(U_max, X_2, "r-", linewidth=1.0)
    plt.plot(U_max, X_3, "b-", linewidth=1.0)
    plt.plot(U_max, X_4, "y-", linewidth=1.0)
    plt.plot(U_max, X_5, "p-", linewidth=1.0)
    plt.title("Moving Mesh by CXF")
    plt.savefig("./blow_up.png")
    plt.show()
