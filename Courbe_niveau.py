import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
N = 1000
eps = 10**(-5)
delta=0.05
###########################################################################

#Quelques fonctions de R2 dans R ou R2


def exemple(x, y):
    return(np.array([x**2 - 1, y**3-2]))

def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return (3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 )

def f2(x1,x2):
    return(np.array([f1(x1,x2) - 0.8, x1 - x2]))

def f3(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2

def f4(x1, x2):
    return(x1**2 + 2*x2 - 3*x1*x2)
###########################################################################
                #Algo de base pour des fonctions de R2 dans R ou dans R2


# Calcul de la jacobienne d'une fonction
def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

#Calcul du gradient d'une fonction
def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f

#Calcul de la tangente à une courbe d'une fonction
def tangente(x, y, f):
    gr = grad(f)(x, y)
    result = np.array([gr[0], -gr[1]])
    norm = np.sqrt(result[0]**2 + result[1]**2) 
    return ((1/norm)*result)



###########################################################################


            #Outils pour le calcul numerique 

#Resolution d'équation par méthode de Newton
def Newton(F, x0, y0, eps=eps, N=N):
    x,y = x0, y0
    J_F = J(F)
    for i in range(N):
        """print(J_F(x,y))"""
        x -= np.dot(np.linalg.inv(J_F(x, y)), F(x, y))[0]
        y -= np.dot(np.linalg.inv(J_F(x, y)), F(x, y))[1]
        
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")


#Calcul d'un  nouveau point sur la courbe de niveau de (x0, y0) à une distance delta
def new_point(f,x0,y0,delta=delta,eps=eps):
    grx,gry=grad(f)(x0,y0)
    direction=np.array([gry, -grx])
    norme=np.sqrt(grx**2+gry**2)
    direction_normalisee=(1/norme)*direction
    c = f(x0, y0)
    def F(a,b):
        return(np.array([f(a,b)-c, (x0-a)**2+(y0-b)**2-delta**2]))
    x , y = x0 + direction_normalisee[0]*delta , y0 + direction_normalisee[1]*delta
    return (Newton(F, x, y, eps, 100))

#Vérifie si deux segments se coupent
def intersec_segm(S1,S2):
    A , B = np.array(S1[0]) , np.array(S1[1])
    C , D = np.array(S2[0]) , np.array(S2[1])
    AB = np.array([B[0]-A[0], B[1]-A[1], 0.])
    CD = np.array([D[0]-C[0], D[1]-C[1], 0.])
    AD = np.array([D[0]-A[0], D[1]-A[1], 0.])
    AC = np.array([C[0]-A[0], C[1]-A[1], 0.])
    CB = np.array([B[0]-C[0], B[1]-A[1], 0.])
    prod1 = np.cross(AB, CD)
    prod2 = np.cross(AB, AD)
    prod3 = np.cross(AB, AC)
    prod4 = np.cross(CD, CB)
    prod5 = np.cross(CD, -AC)
    scal1 = np.vdot(prod2, prod3)
    scal2 = np.vdot(prod4, prod5)
    norm = np.sqrt(np.vdot(prod1,prod1))
    if (A==C).all() or (A==D).all() or (B==C).all() or (B==D).all():
        return(False)
    elif norm !=0 and scal1<=0 and scal2<=0 :
        return (True)
    else :
        return (False) 

#fonction d'interpolation pour relier deux points (avec deux vecteurs tangents au début et à la fin)
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
    print(det)
    if det != 0 :
        
        Y = np.array([x1, y1, x2, y2, 0, 0, 0, 0])
        X = np.linalg.solve(M, Y)
        a , b , c , d , e , f , lamba , mu = X[0] , X[1] , X[2] , X[3] , X[4] , X[5] , X[6] , X[7] 
        
        if lamba >0 and mu >0 :
            
            x_t = a + b*t + c*t*t
            y_t = d + e*t + f*t*t
            return([x_t, y_t])
        else :
            x_t = (1-t)*x1 + t*x2
            y_t = (1-t)*y1 + t*y2
            return([x_t, y_t])
    else :
        x_t = (1-t)*x1 + t*x2
        y_t = (1-t)*y1 + t*y2
        return([x_t, y_t])




###########################################################################




#Courbe de niveau sans condition d'arrêt, juste s'arrête à un nombre de point défini
def level_curve(f, x0, y0, delta=delta, N=N, eps=eps):
    liste_points=np.eye(N, 2)
    liste_points[0][0]=x0
    liste_points[0][1]=y0
    x,y=x0,y0
    for i in range(N-1):
        x,y=new_point(f, x, y, delta, eps)
        liste_points[i+1][0], liste_points[i+1][1]=x,y
    return(liste_points)

#Courbe de niveau qui s'arrête si deux segments s'intersectent ou si un point crée est trop proche d'un autre 
def level_curve_corrig(f, x0, y0, delta=delta, eps=eps):
    liste_points=[[x0, y0]]
    x , y = new_point(f, x0, y0, delta)
    liste_points.append([x,y])
    S1=[[x0, y0], [x, y]]
    S2=[]
    boucle=False
    c=0
    while not boucle :
        x_old , y_old = x , y
        x, y = new_point(f, x, y, delta)
        print(x,y)
        liste_points.append([x, y])
        S2=[[x_old, y_old], [x, y]]
        boucle = intersec_segm(S1, S2)
        if ((x0-x)**2 + (y0-y)**2)<(delta**2) :
            boucle = True
        c+=1
    return (np.array(liste_points))

#Courbe de niveau lissée par une interpolation
def level_curve_complete (f, x0, y0, oversampling, delta=delta, eps=eps):
    if oversampling == 1 :
        return level_curve_corrig(f, x0,y0, delta, eps)
    
    elif oversampling > 1 :
        N = oversampling 
        T = np.linspace(0., 1., N)
        liste_points=[[x0, y0]]
        x , y = new_point(f, x0, y0, delta)
        liste_points.append([x,y])
        S1=[[x0, y0], [x, y]]
        S2=[]
        boucle=False
        c=0 
        u1 = tangente(x0, y0, f)
        u2 = tangente(x, y, f)
        while not boucle :
            x_old , y_old = x , y
            x, y = new_point(f, x, y, delta)
            S2=[[x_old, y_old], [x, y]]
            
            u1 = u2
            u2 = tangente(x, y, f)
            Z = gamma(T, (x_old, y_old), (x, y), u1, u2)
            for i in range(1,len(Z[0])):
                liste_points.append([Z[0][i], Z[1][i]])


            if ((x0-x)**2 + (y0-y)**2)<delta**2 :
                boucle = True
            boucle = intersec_segm(S1, S2)
            c+=1
        return (np.array(liste_points))

###########################################################################

            #Tracée de courbes et sandbox

def display_contour(f, x, y, levels):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    contour_set = plt.contour(
        X, Y, Z, colors="grey", linestyles="dashed", 
        levels=levels 
    )
    ax.clabel(contour_set)
    plt.grid(True)
    plt.xlabel("$x_1$") 
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal")

"""display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)"""
"""X , Y = level_curve_complete(f2, 0.8, 0.8, 10)
plt.scatter(X, Y, color = 'green')"""


"""Z = level_curve_corrig(f1, 1., 1., 5, 0.5)
print(Z)
print(len(Z))
plt.scatter(Z[:,0], Z[:,1])
plt.show()"""
"""Z = gamma(np.linspace(0,1,10), (0.,0.),(1., 1.), (1., 2.), (2.,1.))
print(Z)"""
P_1 = (0.,0.)
P_2 = (5., 5.)
u_1 = (0., 1.)
u_2 = (1.,0.)
Z = gamma(np.linspace(0,1,50), P_1, P_2, u_1, u_2)
X , Y = Z[0], Z[1]
x_1 , y_1 = P_1
x_2 , y_2 = P_2
plt.scatter(X,Y)
plt.scatter(np.array([x_1]), np.array([y_1]), color = 'red')
plt.scatter(np.array([x_2]), np.array([y_2]), color = 'red')

ax = plt.axes()
ax.arrow(P_1[0], P_1[1], u_1[0], u_1[1], head_width=0.1, head_length=0.1, fc='lightblue', ec='black')
ax.arrow(P_2[0], P_2[1], u_2[0], u_2[1], head_width=0.1, head_length=0.1, fc='lightblue', ec='black')

plt.show()
