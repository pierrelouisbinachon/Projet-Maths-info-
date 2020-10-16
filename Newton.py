def Newton(F, x0, y0, eps=eps, N=N):
    ...
    for i in range(N):
        ...
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")