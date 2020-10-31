N = 100
eps = 10**(-10) #totalement choisi au pif
import autograd
import autograd.numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

def Newton(F, x0, y0, eps=eps, N=N):
    x,y = x0, y0
    J_F = J(F)
    for i in range(N):
        print(J_F(x,y))
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
def f3(x1,x2):
    return(np.array([f1(x1,x2) - 0.8, x1-x2 -0.3]))

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
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
x,y = Newton(f2, 0.8, 0.8)

plt.scatter(np.array([x]), np.array([y]))
x,y = Newton(f3, np.random.random(), np.random.random())
plt.scatter(np.array([x]), np.array([y]), color = 'green')

    
plt.show()