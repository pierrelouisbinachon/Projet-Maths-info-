
from Newton import *
import autograd
import autograd.numpy as np
eps = 10**(-20)
def level_curve(f, x0, y0, delta=0.1, N=1000, eps=eps):
    c = f(x0,y0)
    def F(x,y):
        return(np.array([f(x,y) - c, x+y]))
    return(Newton(F,x0, y0, eps, 100))
print(level_curve(f1, 0.44721, 0.44721))