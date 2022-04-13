import numpy as np
import quadpy as qp
from configuration import Config
from functions import norm_coeffs

class Matrices:
    def __init__(self, quad_type=Config.quad_type, n_qp_1d=Config.n_qp_1d, r_max=Config.r_max):
        self.quad_type = quad_type
        self.n_qp_1d = n_qp_1d
        n_qp = n_qp_1d*n_qp_1d
        self.r_max = r_max

        if quad_type == "leg":
        # Gauss-Legendre quadrature
            [pts,wts]=np.polynomial.legendre.leggauss(n_qp_1d)
        elif quad_type == "lob":
        # Gauss-Lobatto quadrature
            scheme=qp.line_segment.gauss_lobatto(n_qp_1d)
            pts=scheme.points
            wts=scheme.weights
        else:
            [pts,wts]=np.polynomial.legendre.leggauss(n_qp_1d)
            print (type,"unsupported quadrature rule, using Legendre")

        self.pts2d_x = np.kron(pts,np.ones(n_qp_1d))
        self.pts2d_y = np.kron(np.ones(n_qp_1d),pts)
        self.wts2d   = np.kron(wts,wts)

        self.phi_val_cell={}
        self.phi_grad_cell_x={}
        self.phi_grad_cell_y={}
        self.phi_val_bd_cell_n={}
        self.phi_val_bd_cell_s={}
        self.phi_val_bd_cell_e={}
        self.phi_val_bd_cell_w={}

        lower  = -np.ones(1)   # Corresponds to the negative boundary of a Legendre polynomial
        upper  =  np.ones(1)   # Corresponds to the positive boundary of a Legendre polynomial
        for r in range(r_max+1):
            num_coeff = r+1

        # Determine the coefficients for the orthogonality
            coeffs = norm_coeffs(num_coeff)
            # TODO just test use above

        # Values and grads of basis functions in internal quadrature points, i.e.
        # phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
            legvander2d = np.polynomial.legendre.legvander2d(self.pts2d_x,self.pts2d_y,[r, r])
            self.phi_val_cell[r] = np.multiply(legvander2d,coeffs)
        # Gradients of the basis functions in internal quadrature points,
        # they has two components due to the x and y derivatives
        # Not a trivial calculation, since gradients of Legendre not supported in numpy
        # and there is probably a much better way to calculate these...
            self.phi_grad_cell_x[r]=np.zeros((n_qp,num_coeff*num_coeff))
            self.phi_grad_cell_y[r]=np.zeros((n_qp,num_coeff*num_coeff))
            temp_vander_x = np.polynomial.legendre.legvander(self.pts2d_x,r)
            temp_vander_y = np.polynomial.legendre.legvander(self.pts2d_y,r)
            dLm_x         = np.zeros((n_qp,num_coeff))
            dLm_y         = np.zeros((n_qp,num_coeff))

            # Note the unclean way with which the coefficients are set, used, and then zeroed out again
            coeff = np.zeros(num_coeff)
            for m in range(num_coeff):
                coeff[m]=1.0
                dLm_x[:,m] = np.polynomial.legendre.legval(self.pts2d_x,np.polynomial.legendre.legder(coeff))
                dLm_y[:,m] = np.polynomial.legendre.legval(self.pts2d_y,np.polynomial.legendre.legder(coeff)) 
                coeff[m]=0.0

            for m in range(num_coeff):
                for n in range(num_coeff):
                    self.phi_grad_cell_x[r][:,m*num_coeff+n]=np.multiply(temp_vander_y[:,n],dLm_x[:,m])
                    self.phi_grad_cell_y[r][:,m*num_coeff+n]=np.multiply(temp_vander_x[:,m],dLm_y[:,n])

            self.phi_grad_cell_x[r] = np.multiply(self.phi_grad_cell_x[r],coeffs)
            self.phi_grad_cell_y[r] = np.multiply(self.phi_grad_cell_y[r],coeffs)

        # Values of basis functions in boundary quadrature points, repeating for
        # each face and degree;  dimensions: (num_quad_pts_per_face)x(cardinality)x(num_faces)
            self.phi_val_bd_cell_n[r] = np.multiply(np.polynomial.legendre.legvander2d(pts,   upper, [r,r]),coeffs)
            self.phi_val_bd_cell_s[r] = np.multiply(np.polynomial.legendre.legvander2d(pts,   lower, [r,r]),coeffs)
            self.phi_val_bd_cell_e[r] = np.multiply(np.polynomial.legendre.legvander2d(upper, pts,   [r,r]),coeffs)
            self.phi_val_bd_cell_w[r] = np.multiply(np.polynomial.legendre.legvander2d(lower, pts,   [r,r]),coeffs)