import numpy as np

# Function to compute mass matrix and its inverse
def compute_mass(phi,wts2d,nx,ny,r,hx,hy,y_c,pts2d_y,eq_type) :
    n_qp = len(pts2d_y)
    # determ  = hx*hy/4
    # mass = np.zeros((nx, ny))
    # cos_factor = np.zeros((nx, ny))
    # sin_factor = np.zeros((nx, ny))
    # inv_mass = np.zeros((nx, ny))

    cos_factor = np.ones((nx, ny, 1, n_qp))
    sin_factor = np.zeros((nx, ny, 1, n_qp))

    for j in range(ny):
        if (eq_type == "linear" or eq_type == "swe") :
            # cos_factor = np.ones((n_qp, n_qp))
            pass
        elif (eq_type == "adv_sphere" or eq_type == "swe_sphere") :
            cos_y = np.cos( y_c[j]+pts2d_y*hy/2 ) 
            sin_y = np.sin( y_c[j]+pts2d_y*hy/2 ) 
            cos_factor[:,j,0,:] = np.tile(cos_y, (nx, 1))
            sin_factor[:,j,0,:] = np.tile(sin_y, (nx, 1))
        else :
            print (eq_type,"unsupported option, using linear ")
            cos_factor[j] = np.ones((n_qp, n_qp))

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

    return mass, inv_mass