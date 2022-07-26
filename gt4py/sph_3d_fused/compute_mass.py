import numpy as np
import boundary_conditions

# Function to compute mass matrix and its inverse
def compute_mass(phi,wts2d,nx,ny,r,hx,hy,y_c,pts2d_y,pts,eq_type) :
    n_qp = len(pts2d_y)
    n_qp_1D =  int(np.sqrt(n_qp))
    # determ  = hx*hy/4
    # mass = np.zeros((nx, ny))
    # cos_factor = np.zeros((nx, ny))
    # sin_factor = np.zeros((nx, ny))
    # inv_mass = np.zeros((nx, ny))

    # Internal
    cos_factor = np.ones((ny, n_qp))
    sin_factor = np.zeros((ny, n_qp))
    # Boundary
    cos_n = np.zeros((ny, n_qp_1D))
    cos_s = np.zeros((ny, n_qp_1D))

    for j in range(ny):
        cos_factor[j] = np.cos(y_c[j]+pts2d_y*hy/2) 
        sin_factor[j] = np.sin(y_c[j]+pts2d_y*hy/2) 
        cos_n[j] = np.cos(y_c[j] + hy/2)
        cos_s[j] = np.cos(y_c[j] - hy/2)


    dim = (r+1)**2
    determ  = hx*hy/4
    mass = np.zeros((ny, dim, dim))
    inv_mass = np.zeros((ny, dim, dim))
    for j in range(ny):
        cos_term = cos_factor[j]
        # r_loc=rdist[i,j]
        # dim = int((r_loc+1)*(r_loc+1))
        matrix = np.zeros((dim,dim))
        for m in range(dim):
            for n in range(dim):
                # det*sum(i-th basis function in qp * j-th basis function in qp * metric factor * weights)
                # matrix[m,n]=determ * np.dot(wts2d, (phi[r_loc][:,m] * phi[r_loc][:,n] * cos_factor[j]))
                matrix[m,n]=determ * np.dot(wts2d, (phi[:,m] * phi[:,n] * cos_term))
        mass[j] = matrix
        inv_mass[j] = np.linalg.inv(mass[j])

    return mass, inv_mass, cos_factor, sin_factor, cos_n, cos_s