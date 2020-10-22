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

"""point = new_point(f1,0.3,0.6,10**(-1),eps)
print(point)
print(f1(0.3, 0.6))

print(f1(point[0], point[1]))"""

def level_curve(f, x0, y0, delta=0.1, N=100, eps=eps):
    liste_points=np.eye(N, 2)
    liste_points[0][0]=x0
    liste_points[0][1]=y0
    x,y=x0,y0
    for i in range(N-1):
        x,y=new_point(f, x, y, delta, eps)
        liste_points[i+1][0], liste_points[i+1][1]=x,y
    return(liste_points)

print(level_curve(f1, 5., 6.))


def intersec_segm(S1,S2):
    A , B = np.array(S1[0]) , np.array(S1[1])
    C , D = np.array(S2[0]) , np.array(S2[1])
    V1 = np.array([B[0]-A[0], B[1]-A[1], 0.])
    V2 = np.array([D[0]-C[0], D[1]-C[1], 0.])
    V3 = np.array([D[0]-A[0], D[1]-A[1], 0.])
    V4 = np.array([C[0]-A[0], C[1]-A[1], 0.])
    V5 = np.array([B[0]-C[0], B[1]-A[1], 0.])
    prod1 = np.cross(V1, V2)
    prod2 = np.cross(V1, V3)
    prod3 = np.cross(V1, V4)
    prod4 = np.cross(V2, V5)
    prod5 = np.cross(V2, -V4)
    scal1 = np.vdot(prod2, prod3)
    scal2 = np.vdot(prod4, prod5)
    norm = np.sqrt(np.vdot(prod1,prod1))
    if norm !=0 and scal1<=0 and scal2<=0 :
        return (True)
    return (False) 

"""print(intersec_segm([[0.,-100.],[0.,100.]],[[-5.,5.],[7.,8.]]))"""





