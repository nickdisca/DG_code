import numpy as np
def compute_unif(a, b, n, rdist=None):

    x_e, hx = np.linspace(a, b, n+1, retstep=True)
    
    x = [edge + hx * np.linspace(0, 1, 4, endpoint=False) for edge in x_e[:-1]]; x.append(np.array(b).reshape(1,))
    return np.concatenate(x)


