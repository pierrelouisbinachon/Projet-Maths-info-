import autograd
import autograd.numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
N = 100
eps = 10**(-10)

###########################################################################

#Quelques fonctions de R2 dans R ou R2


def exemple(x, y):
    return(np.array([x**2 - 1, y**3-2]))

def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 

def f2(x1,x2):
    return(np.array([f1(x1,x2) - 0.8, x1 - x2]))



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
    return (result)



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
def new_point(f,x0,y0,delta,eps=eps):
    grx,gry=grad(f)(x0,y0)
    direction=np.array([gry, -grx])
    norme=np.sqrt(grx**2+gry**2)
    direction_normalisee=(1/norme)*direction
    c = f(x0, y0)
    def F(a,b):
        return(np.array([f(a,b)-c, (x0-a)**2+(y0-b)**2-delta**2]))
    x , y = x0 + direction_normalisee[0]*delta , y0 + direction_normalisee[1] *delta
    return (Newton(F, x, y, eps, 100))

#Vérifie si deux segments se coupent
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

def Intersect(S1,S2):
    P1 = S1[0]
    P2, P3, P4 = S1[1], S2[0], S2[1]
    if (max(P1[0],P2[0]) < min(P3[0],P4[0])): # Pas d'intersection possible car intervalles disjoints.
        return False

    if P1[0]-P2[0] != 0 and P3[0]-P4[0] !=0 :
        A1 = (P1[1]-P2[1])/(P1[0]-P2[0])  # y = ax+b
        A2 = (P3[1]-P4[1])/(P3[0]-P4[0])
        b1 = P1[1]-A1*P1[0]
        b2 = P3[1]-A2*P3[0]
        if A1==A2 : # Segments parrallèles
            return(False)
        else : 
            Xa = (b2 - b1) / (A1 - A2) ## abscisse de l'intersection
            if ( (Xa < max( min(P1[0],P2[0]), min(P3[0],P4[0]) )) or (Xa > min( max(P1[0],P2[0]), max(P3[0],P4[0]) )) ):
                return False #L'abscisse de l'intersection doit être dans le domaine de définition des segments
            else : 
                return True
    else : #pas utile avec approx flottant 
        if (P1[0]-P2[0]==0 and P3[0]-P4[0]==0):
            return False
        else :
            if P1[0]-P2[0]==0 : # on a x = c
                X = P1[0]
                Y =  (P3[1]-P4[1])/(P3[0]-P4[0])*P1[0] + P3[1]-A2*P3[0]
            else :
                X = P3[0]
                Y = (P1[1]-P2[1])/(P1[0]-P2[0])*P3[0]+ P1[1]-A1*P1[0]
            if (min(P1[0],P2[0])<= X<= max(P1[0],P2[0]) and min(P3[0],P4[0])<= X<= max(P3[0],P4[0])) and (min(P1[1],P2[1])<= Y<= max(P1[0],P2[0]) and min(P3[1],P4[1])<= Y<= max(P3[1],P4[1])):
                return True
            else :
                return False
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
    
    if det != 0 :
        
        Y = np.array([x1, y1, x2, y2, 0, 0, 0, 0])
        X = np.linalg.solve(M, Y)
        a , b , c , d , e , f , lamba , mu = X[0] , X[1] , X[2] , X[3] , X[4] , X[5] , X[6] , X[7] 
        
        if lamba >0 and mu >0 :
            
            x_t = a + b*t + c*t*t
            y_t = d + e*t + f*t*t
            return((x_t, y_t))
        else :
            x_t = (1-t)*x1 + t*x2
            y_t = (1-t)*y1 + t*y2
            return(x_t, y_t)

    else :
        x_t = (1-t)*x1 + t*x2
        y_t = (1-t)*y1 + t*y2
        return(x_t, y_t)




###########################################################################




#Courbe de niveau sans condition d'arrêt, juste s'arrête à un nombre de point défini
def level_curve(f, x0, y0, delta=0.1, N=100, eps=eps):
    liste_points=np.eye(N, 2)
    liste_points[0][0]=x0
    liste_points[0][1]=y0
    x,y=x0,y0
    for i in range(N-1):
        x,y=new_point(f, x, y, delta, eps)
        liste_points[i+1][0], liste_points[i+1][1]=x,y
    return(liste_points)

#Courbe de niveau qui s'arrête si deux segments s'intersectent ou si un point crée est trop proche d'un autre 
def level_curve_corrig(f, x0, y0, delta=0.1, eps=eps):
    liste_points=[[x0, y0]]
    x , y = new_point(f, x0, y0, delta)
    liste_points.append([x,y])
    S1=[[x0, y0], [x, y]]
    print(S1)
    S2=[]
    
    x_old , y_old = x , y
    x, y = new_point(f, x, y, delta)
    liste_points.append([x, y])
    boucle = False
    while not boucle:
        
        x_old , y_old = x , y
        x, y = new_point(f, x, y, delta)
        
        
        liste_points.append([x, y])
        S2=[[x_old, y_old], [x, y]]
        boucle = Intersect(S1, S2)
        
    return (np.array(liste_points))

#print(intersec_segm([[0.1, 0.1], [0.09733088242259676, 0.19996437271027112]],[[0.10455034318514403, -0.4817716355656937], []]))
#Courbe de niveau lissée par une interpolation
def level_curve_complete (f, x0, y0, oversampling, delta=0.1, eps=eps):
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
        N = 0
        while not boucle and N<100:
            N += 1
            x_old , y_old = x , y
            x, y = new_point(f, x, y, delta)
            
            u1 = tangente(x_old, y_old, f)
            u2 = tangente(x, y, f)
            
            X , Y = gamma(T, (x_old, y_old), (x, y), u1, u2)
            
            for i in range(1,len(X)):
                liste_points.append([X[i], Y[i]])

            S2=[[x_old, y_old], [x, y]]
            boucle = intersec_segm(S1, S2)
        return(liste_points)
def F(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2
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
display_contour(
    F, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
Z = level_curve_corrig(F, 0.1, 0.1)

X = Z[:,0]
Y = Z[:,1]
plt.scatter(X, Y)
plt.show()
