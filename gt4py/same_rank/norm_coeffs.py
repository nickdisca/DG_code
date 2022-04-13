import numpy as np

def norm_coeffs(r):
    result = np.zeros(r*r)
    ind = 0
    for k1 in range(r):
        for k2 in range(r):
            result[ind] = np.sqrt((2*k1+1)*(2*k2+1)) / 2.0
            ind += 1
    return result