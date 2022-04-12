import numpy as np
import numpy.polynomial.legendre as L


class Vander:
    def __init__(self, dim, r_max, n_qp, pts2d_x, pts2d_y):
        self.unif2d_x = []
        self.unif2d_y = []

        self.V = np.zeros((dim, dim, r_max+1))
        self.phi_val_cell = np.zeros((n_qp, dim, r_max+1))
        self.phi_val_cell_x = np.zeros((n_qp, dim, r_max+1))
        self.phi_val_cell_y = np.zeros((n_qp, dim, r_max+1))
        for r in range(r_max+1):
            unif = np.linspace(-1, 1, r+1)
            self.unif2d_x.append(np.kron(unif,np.ones(r+1)))
            self.unif2d_y.append(np.kron(np.ones(r+1),unif))

            num_coeff = r+1
            matrix_dim = num_coeff**2
            # Determine the coefficients for the orthogonality
            coeffs = self.norm_coeffs(num_coeff)

            # Square matrix for the modal-nodal transformations
            self.V[:matrix_dim,:matrix_dim,r]= L.legvander2d(self.unif2d_x[r],self.unif2d_y[r],[r,r])
            # self.V[:matrix_dim,:matrix_dim,r] = legvander2d * coeffs

            # Values and grads of basis functions in internal quadrature points, i.e.
            # phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
            self.phi_val_cell[:, :matrix_dim, r]= np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[r, r])
            # self.phi_val_cell[:, :matrix_dim, r] = legvander2d * coeffs

            



    def norm_coeffs(self, r):
        result = np.zeros(r*r)
        ind = 0
        for k1 in range(r):
            for k2 in range(r):
                result[ind] = np.sqrt((2*k1+1)*(2*k2+1)) / 2.0
                ind += 1
        return result  