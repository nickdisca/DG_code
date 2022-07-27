import numpy as np

# Function to compute mass matrix and its inverse
def compute_mass(phi,wts2d,nx,ny,r,hx,hy,y_c,pts2d_y,eq_type) :
    n_qp = len(pts2d_y)
    cos_factor = np.ones((nx, ny, 1, n_qp))
    dim = (r+1)**2
    determ  = hx*hy/4
    mass = np.zeros((nx, ny, 1, dim, dim))
    inv_mass = np.zeros((nx, ny, 1, dim, dim))
    for j in range(ny):
        for i in range(nx):
            matrix = np.zeros((dim,dim))
            for m in range(dim):
                for n in range(dim):
                    # det*sum(i-th basis function in qp * j-th basis function in qp * metric factor * weights)
                    matrix[m,n]=determ * np.dot(wts2d, (phi[:,m] * phi[:,n] * cos_factor[i,j,0]))
            mass[i,j,0,:,:] = matrix
            inv_mass[i,j,0,:,:] = np.linalg.inv(mass[i,j])

    return mass, inv_mass