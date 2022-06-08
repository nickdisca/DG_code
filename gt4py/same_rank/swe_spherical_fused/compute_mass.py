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
    cos_factor = np.ones((nx, ny, 1, n_qp))
    sin_factor = np.zeros((nx, ny, 1, n_qp))
    # Boundary
    cos_n = np.zeros((nx+2, ny+2, 1, n_qp_1D))
    cos_s = np.zeros((nx+2, ny+2, 1, n_qp_1D))
    cos_e = np.zeros((nx+2, ny+2, 1, n_qp_1D))
    cos_w = np.zeros((nx+2, ny+2, 1, n_qp_1D))

    for j in range(ny):
        cos_y = np.cos(y_c[j]+pts2d_y*hy/2) 
        sin_y = np.sin(y_c[j]+pts2d_y*hy/2) 
        cos_n_y = np.cos(y_c[j] + hy/2)
        cos_s_y = np.cos(y_c[j] - hy/2)
        cos_e_y = np.cos(y_c[j] + pts * hy/2)
        cos_w_y = np.cos(y_c[j] + pts * hy/2)

        cos_factor[:,j,0,:] = np.tile(cos_y, (nx, 1))
        sin_factor[:,j,0,:] = np.tile(sin_y, (nx, 1))
        cos_n[1:nx+1,j+1,0,:] = np.tile(cos_n_y, (nx, 1))
        cos_s[1:nx+1,j+1,0,:] = np.tile(cos_s_y, (nx, 1))
        cos_e[1:nx+1,j+1,0,:] = np.tile(cos_w_y, (nx, 1))
        cos_w[1:nx+1,j+1,0,:] = np.tile(cos_e_y, (nx, 1))

        boundary_conditions.apply_pbc(cos_n)
        boundary_conditions.apply_pbc(cos_s)
        boundary_conditions.apply_pbc(cos_e)
        boundary_conditions.apply_pbc(cos_w)




    dim = (r+1)**2
    determ  = hx*hy/4
    mass = np.zeros((nx, ny, 1, dim, dim))
    inv_mass = np.zeros((nx, ny, 1, dim, dim))
    for j in range(ny):
        for i in range(nx):
            # r_loc=rdist[i,j]
            # dim = int((r_loc+1)*(r_loc+1))
            matrix = np.zeros((dim,dim))
            for m in range(dim):
                for n in range(dim):
                    # det*sum(i-th basis function in qp * j-th basis function in qp * metric factor * weights)
                    # matrix[m,n]=determ * np.dot(wts2d, (phi[r_loc][:,m] * phi[r_loc][:,n] * cos_factor[j]))
                    matrix[m,n]=determ * np.dot(wts2d, (phi[:,m] * phi[:,n] * cos_factor[i,j,0]))
            mass[i,j,0,:,:] = matrix
            inv_mass[i,j,0,:,:] = np.linalg.inv(mass[i,j])

    return mass, inv_mass, cos_factor, sin_factor, cos_n, cos_s, cos_e, cos_w