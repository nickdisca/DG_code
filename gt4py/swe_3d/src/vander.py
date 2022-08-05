import numpy as np
import numpy.polynomial.legendre as L
from scipy.special import legendre
import gt4py as gt


class Vander:
    def __init__(self, nx, ny, nz, dim, r, n_qp, pts2d_x, pts2d_y, pts, wts2d, backend="gtc:numpy"):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.phi_val_cell = np.zeros((n_qp, dim))
        self.phi_grad_cell_x = np.zeros((n_qp, dim))
        self.phi_grad_cell_y = np.zeros((n_qp, dim))
        unif = np.linspace(-1, 1, r+1)
        self.unif2d_x = np.kron(unif,np.ones(r+1))
        self.unif2d_y = np.kron(np.ones(r+1),unif)

        self.wts2d = wts2d
        self.pts = pts


        lower  = -np.ones(1)   # Corresponds to the negative boundary of a Legendre polynomial
        upper  =  np.ones(1)   # Corresponds to the positive boundary of a Legendre polynomial

        num_coeff = r+1
        matrix_dim = num_coeff**2
        # # Determine the coefficients for the orthogonality
        coeffs = self.norm_coeffs(num_coeff)

        # Square matrix for the modal-nodal transformations
        self.vander = L.legvander2d(self.unif2d_x,self.unif2d_y,[r,r])
        self.vander = self.vander * coeffs
        self.inv_vander = np.linalg.inv(self.vander)

        # Values and grads of basis functions in internal quadrature points, i.e.
        # phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
        self.phi_val_cell = np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[r, r])
        self.phi_val_cell = self.phi_val_cell * coeffs

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
                self.phi_grad_cell_x[:,m*num_coeff+n]=np.multiply(temp_vander_y[:,n],dLm_x[:,m])
                self.phi_grad_cell_y[:,m*num_coeff+n]=np.multiply(temp_vander_x[:,m],dLm_y[:,n])
        self.phi_grad_cell_x = self.phi_grad_cell_x * coeffs
        self.phi_grad_cell_y = self.phi_grad_cell_y * coeffs

        # Values of basis functions in boundary quadrature points, repeating for
        # each face and degree;  dimensions: (num_quad_pts_per_face)x(cardinality)x(num_faces)
        self.phi_val_bd_cell_n = np.polynomial.legendre.legvander2d(pts,   upper, [r,r]) * coeffs
        self.phi_val_bd_cell_s = np.polynomial.legendre.legvander2d(pts,   lower, [r,r]) * coeffs
        self.phi_val_bd_cell_e = np.polynomial.legendre.legvander2d(upper, pts,   [r,r]) * coeffs
        self.phi_val_bd_cell_w = np.polynomial.legendre.legvander2d(lower, pts,   [r,r]) * coeffs

        self.conv_to_gt(backend=backend)


    def conv_to_gt(self, backend, dtype=np.float64):
        dim = self.vander.shape[0]
        n_qp_1D = len(self.pts)
        n_qp, dim = self.phi_val_cell.shape

        self.wtsd2d_gt = gt.storage.from_array(data=self.wts2d,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp,)), default_origin=(0,0,0))
        self.vander_gt = gt.storage.from_array(data=self.vander,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (dim, dim)), default_origin=(0,0,0))
        self.inv_vander_gt = gt.storage.from_array(data=self.inv_vander,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (dim, dim)), default_origin=(0,0,0))

        self.phi_gt = gt.storage.from_array(data=self.phi_val_cell,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp, dim)), default_origin=(0,0,0))
        self.grad_phi_x_gt = gt.storage.from_array(data=self.phi_grad_cell_x,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp, dim)), default_origin=(0,0,0))
        self.grad_phi_y_gt = gt.storage.from_array(data=self.phi_grad_cell_y,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp, dim)), default_origin=(0,0,0))

        self.phi_bd_N_gt = gt.storage.from_array(data=self.phi_val_bd_cell_n,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp_1D,dim)), default_origin=(0,0,0))
        self.phi_bd_S_gt = gt.storage.from_array(data=self.phi_val_bd_cell_s,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp_1D,dim)), default_origin=(0,0,0))
        self.phi_bd_E_gt = gt.storage.from_array(data=self.phi_val_bd_cell_e,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp_1D,dim)), default_origin=(0,0,0))
        self.phi_bd_W_gt = gt.storage.from_array(data=self.phi_val_bd_cell_w,
            backend=backend, shape=(self.nx, self.ny, self.nz), dtype = (dtype, (n_qp_1D,dim)), default_origin=(0,0,0))
        

    def norm_coeffs(self, r):
        result = np.zeros(r*r)
        ind = 0
        for k1 in range(r):
            for k2 in range(r):
                result[ind] = np.sqrt((2*k1+1)*(2*k2+1)) / 2.0
                ind += 1
        return result  