from Newton import *
import autograd
import autograd.numpy as np
import matplotlib.pyplot as pl
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

"""print(level_curve(f1, 5., 6.))"""


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
    if (A==C).all() or (A==D).all() or (B==C).all() or (B==D).all():
        return(False)
    elif norm !=0 and scal1<=0 and scal2<=0 :
        return (True)
    else :
        return (False) 

"""print(intersec_segm([[0.,-100.],[0.,100.]],[[-5.,5.],[7.,8.]]))"""

def level_curve_corrig(f, x0,y0, delta=0.1, eps=eps):
    liste_points=[[x0, y0]]
    x , y = new_point(f, x0, y0, delta)
    liste_points.append([x,y])
    S1=[[x0, y0], [x, y]]
    S2=[]
    boucle=False
    while not boucle :
        x_old , y_old = x , y
        x, y = new_point(f, x, y, delta)
        liste_points.append([x, y])
        S2=[[x_old, y_old], [x, y]]
        boucle = intersec_segm(S1, S2)
    return (np.array(liste_points))


"""print(level_curve_corrig(f1, 1. , 2. ,))      """  

def gamma(t, P1, P2, u1, u2) :
    
    x1 , y1 = P1[0] , P1[1]
    x2 , y2 = P2[0] , P2[1]
    alpha1 , beta1 = u1[0] , u1[1]
    alpha2 , beta2 = u2[0] , u2[1]
    
    M=np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0], 
    [1, 1, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, -alpha1, 0], 
    [0, 0, 0, 0, 1, 0, -beta1, 0],
    [0, 1, 2, 0, 0, 0, 0, -alpha2],
    [0, 0, 0, 0, 1, 2, 0, -beta2]])
    
    det = np.linalg.det(M)
    
    if det != 0 :
        
        Y = np.array([x1, y1, x2, y2, 0, 0, 0, 0])
        X = np.linalg.solve(M, Y)
        a , b , c , d , e , f , lamba , mu = X[0] , X[1] , X[2] , X[3] , X[4] , X[5] , X[6] , X[7] 
        
        if lamba >0 and mu >0 :
            
            x_t = a + b*t + c*t*t
            y_t = d + e*t + f*t*t
            return((x_t, y_t))
    else :
        return(none)

T = np.linspace(0., 1., 10)
print(gamma(T, (0.,0.), (1.,1.), (1.,4.), (2., -6.)))






