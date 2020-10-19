N = 100
eps = 10**(-10) #totalement choisi au pif
import autograd
import autograd.numpy as np

def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

def Newton(F, x0, y0, eps=eps, N=N):
    x,y = x0, y0
    J_F = J(F)
    for i in range(N):
        x -= np.dot(np.linalg.inv(J_F(x, y)), F(x, y))[0]
        y -= np.dot(np.linalg.inv(J_F(x, y)), F(x, y))[1]
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

def exemple(x, y):
    return(np.array([x**2 - 1, y**3-2]))



#tache 2:

def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 

def f2(x1,x2):
    return(np.array([f1(x1,x2) - 0.8, x1 - x2]))

