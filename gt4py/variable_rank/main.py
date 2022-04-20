# %%
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
import quadpy as qp

from degree_distribution import degree_distribution
from compute_unif import compute_unif
from norm_coeffs import norm_coeffs

from numpy.polynomial import legendre as L

from vander import Vander

# %%

# %%
# Radius of the earth (for spherical geometry)
radius=6.37122e6;

# Equation type
eq_type="adv_sphere";

# domain
a = 0; b = 1; c = 0; d =1
# number of elements in X and Y
nx = 20; ny = 20

hx = (b-a)/nx; hy = (d-c)/ny

# polynomial degree of DG
r_max = 2
# cardinality
dim=(r_max+1)**2

# Type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg"

# Number of quadrature points in one dimension
n_qp_1D=4

# Number of quadrature points
n_qp=n_qp_1D*n_qp_1D
# %%
rdist_gt = degree_distribution("unif",nx,ny,r_max);

if quad_type == "leg":
# Gauss-Legendre quadrature
    [pts,wts]=np.polynomial.legendre.leggauss(n_qp_1D)
elif quad_type == "lob":
# Gauss-Lobatto quadrature
    scheme=qp.line_segment.gauss_lobatto(n_qp_1D)
    pts=scheme.points
    wts=scheme.weights
else:
    [pts,wts]=np.polynomial.legendre.leggauss(n_qp_1D)
    print (type,"unsupported quadrature rule, using Legendre")

pts2d_x = np.kron(pts,np.ones(n_qp_1D))
pts2d_y = np.kron(np.ones(n_qp_1D),pts)
wts2d   = np.kron(wts,wts)

half_cell_x = (b-a)/(2*nx)
half_cell_y = (d-c)/(2*ny)
x_c=np.linspace(a+half_cell_x,b-half_cell_x,nx); # Cell centers in X
y_c=np.linspace(c+half_cell_y,d-half_cell_y,ny); # Cell centers in Y

# To support the variable length of the uniform space, we use lists
unif2d_x = {}
unif2d_y = {}

# The Kronecker product is used to form the tensor
for r in range(r_max+1):
     unif=np.linspace(-1,1,r+1)
     unif2d_x[r] = np.kron(unif,np.ones(r+1))
     unif2d_y[r] = np.kron(np.ones(r+1),unif)

# all matrices are the same size but lower orders are padded!
V = np.zeros((dim, dim, r_max+1))
phi_val_cell = np.zeros((n_qp, dim, r_max+1))

V = Vander(dim, r_max, n_qp, pts2d_x, pts2d_y)

print('Done')

# for r in range(r_max+1):
#     num_coeff = r+1
#     matrix_dim = (r+1)**2

#     # Determine the coefficients for the orthogonality
#     coeffs = norm_coeffs(num_coeff)

#     # Square matrix for the modal-nodal transformations
#     legvander2d = L.legvander2d(unif2d_x[r],unif2d_y[r],[r,r])
#     V[:matrix_dim,:matrix_dim,r] = legvander2d * coeffs

#     # Values and grads of basis functions in internal quadrature points, i.e.
#     # phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
#     legvander2d = np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[r, r])
#     phi_val_cell[:, :matrix_dim, r] = legvander2d * coeffs
