import numpy as np
import numpy.polynomial.legendre as L
from scipy.special import legendre


class Vander:
    def __init__(self, dim, r_max, n_qp, pts2d_x, pts2d_y):
        # self.V = np.zeros((dim, dim))
        self.phi_val_cell = np.zeros((n_qp, dim))
        self.phi_grad_cell_x = np.zeros((n_qp, dim))
        self.phi_grad_cell_y = np.zeros((n_qp, dim))
        unif = np.linspace(-1, 1, r_max+1)
        self.unif2d_x = np.kron(unif,np.ones(r_max+1))
        self.unif2d_y = np.kron(np.ones(r_max+1),unif)

        num_coeff = r_max+1
        matrix_dim = num_coeff**2
        # # Determine the coefficients for the orthogonality
        # coeffs = self.norm_coeffs(num_coeff)

        # Square matrix for the modal-nodal transformations
        self.V = L.legvander2d(self.unif2d_x[r_max],self.unif2d_y[r_max],[r_max,r_max])
        # self.V[:matrix_dim,:matrix_dim,r] = legvander2d * coeffs

        # Values and grads of basis functions in internal quadrature points, i.e.
        # phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
        self.phi_val_cell[r_max] = np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[r, r])
        # self.phi_val_cell[:, :matrix_dim, r] = legvander2d * coeffs

        temp_vander_x = np.polynomial.legendre.legvander(pts2d_x,r)
        temp_vander_y = np.polynomial.legendre.legvander(pts2d_y,r)
        dLm_x         = np.zeros((n_qp,num_coeff))
        dLm_y         = np.zeros((n_qp,num_coeff))

        # Note the unclean way with which the coefficients are set, used, and then zeroed out again
        coeff = np.zeros(num_coeff)
        for m in range(num_coeff):
            coeff[m]=1.0
            dLm_x[:,m] = np.polynomial.legendre.legval(pts2d_x,np.polynomial.legendre.legder(coeff))
            dLm_y[:,m] = np.polynomial.legendre.legval(pts2d_y,np.polynomial.legendre.legder(coeff)) 
            coeff[m]=0.0

        for m in range(num_coeff):
            for n in range(num_coeff):
                self.phi_grad_cell_x[:,m*num_coeff+n, r]=np.multiply(temp_vander_y[:,n],dLm_x[:,m])
                self.phi_grad_cell_y[:,m*num_coeff+n, r]=np.multiply(temp_vander_x[:,m],dLm_y[:,n])

        print('Intilization finished!')





    def norm_coeffs(self, r):
        result = np.zeros(r*r)
        ind = 0
        for k1 in range(r):
            for k2 in range(r):
                result[ind] = np.sqrt((2*k1+1)*(2*k2+1)) / 2.0
                ind += 1
        return result  