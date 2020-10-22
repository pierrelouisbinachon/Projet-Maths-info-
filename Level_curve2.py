from Newton import *
import autograd
import autograd.numpy as np
eps=10**(-10)

def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f

def new_point(f,x0,y0,delta,eps=eps):
    grx,gry=grad(f)(x0,y0)
    direction=np.array([-gry,grx])
    norme=np.sqrt(grx**2+gry**2)
    direction_normalisee=(1/norme)*direction
    c = f(x0, y0)
    def F(a,b):
        return(np.array([f(a,b)-c, (x0-a)**2+(y0-b)**2-delta**2]))
    x , y = x0 + direction_normalisee[0] , y0 + direction_normalisee[1]
    return (Newton(F, x, y, eps, 100))


point = new_point(f1,0.3,0.6,10**(-1),eps)
print(point)
print(f1(0.3, 0.6))

print(f1(point[0], point[1]))

#Q8
def gammma(t, P1, P2, u1, u2):
    if u1[0] + u2[0] == 2*(P2[0] - P1[0]) and u1[1] + u2[1] == 2*(P2[1] - P1[1]):
        gamma1 = P1[0]*np.ones(len(t)) + u1[0]*t + (u2[0] - u1[0])/2 *t*t
        gamma2 = P1[1]*np.ones(len(t)) + u1[1]*t + (u2[1] - u1[1])/2 *t*t
        gamma = np.concatenate(gamma1, gamma2)
        gamma.reshape(2,len(t))
        return(gamma)