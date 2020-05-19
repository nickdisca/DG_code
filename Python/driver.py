from dataclasses import dataclass
import numpy as np
import quadpy as qp
import math

# definition of the domain [a,b]x[c,d]
a=0; b=2*math.pi; c=-math.pi/2; d=math.pi/2;

# Function to compute the element degrees
def degree_distribution(type,d1,d2,r_max):
    if type == "unif":
        result = r_max*np.ones((d1, d2))
        print(result)
    elif type == "y_dep":
# Implement:  round( (r_max-1)/(floor(d2/2)-1)*(0:floor(d2/2)-1)+1 );
        result = r_max*np.ones((d1, d2))
    elif type == "y_incr":
# Implement: round ( (r_max-1)/(d2-1)*(0:d2-1)+1 );
        result = r_max*np.ones((d1, d2))
    else: 
        result = r_max*np.ones((d1, d2))
        print (type,"unsupported degree distribution, using uniform")
    return result

# Function to interpolate, generally from modal to nodal space
def modal2nodal(d1,d2,uM,V,r):
    result = {}
    for i in range(d1):
        for j in range(d2):
            result[i,j] =  V[r(i,j)]*uM[i,j]
    return result

# Function to interpolate, generally from nodal to modal space
def modal2nodal(d1,d2,uM,V,r):
    result = {}
    for i in range(d1):
        for j in range(d2):
            result[i,j] =  numpy.linalg.solve(V[r(i,j)],uM[i,j])
    return result


# number of elements in x and y direction
d1=20; d2=20;

# length of the 1D intervals
hx=(b-a)/d1; hy=(d-c)/d2;

# polynomial degree of DG
r_max=2;

# cardinality
dim=(r_max+1)^2;

# Calculate the degrees of the individual elements (np array)
r = degree_distribution("unif",d1,d2,r_max);

# Equation type
eq_type="adv_sphere";

# Type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg";

# Number of quadrature points in one dimension
n_qp_1D=4;

# Number of quadrature points
n_qp=n_qp_1D^2;

# Time interval, initial and final time
t=0;
T=5*86400;

# Order of the RK scheme (1,2,3,4)
RK=4;

# Time step
# For "linadv":  dt=1/r_max^2*min(hx,hy)*0.1;
# For "adv_sphere" with earth radius
dt=100;

# Plotting frequency
plot_freq=100;

half_cell_x = (b-a)/d1/2;
half_cell_y = (d-c)/d2/2;
x_c=np.linspace(a+half_cell_x,b-half_cell_x,d1); # Cell centers in X
y_c=np.linspace(c+half_cell_y,d-half_cell_y,d2); # Cell centers in Y

print(x_c)

print(y_c)

# To support the variable length of the uniform space, we use lists
#
unif2d_x = {}
unif2d_y = {}

#
# The Kronecker product is used to form the tensor
for r in range(r_max):
     unif=np.linspace(-1,1,r+2)
     unif2d_x[r] = np.kron(unif,np.ones(r+2))
     unif2d_y[r] = np.kron(np.ones(r+2),unif)

#
# These are the uniform points used for visualization -- use number of quadrature points

unif=np.linspace(-1,1,n_qp_1D)
unif2d_visual_x = np.kron(unif,np.ones(n_qp_1D))
unif2d_visual_y = np.kron(np.ones(n_qp_1D),unif)

# Again, we use lists for the main fields.  This is not supposed to be efficient
K = {}
for i in range(d1):
    for j in range(d2):
        K[i, j] = np.zeros((3, 3))

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

#
# The Kronecker product is used to form the tensor of quadrature points
pts2d_x = np.kron(pts,np.ones(n_qp_1D))
pts2d_y = np.kron(np.ones(n_qp_1D),pts)
wts2d_y = np.kron(wts,wts)

# Create the Vandermonde matrix for the modal to nodal conversion
V = {}
for r in range(r_max):
    max_deg = (r+1)
    V[r] = np.polynomial.legendre.legvander2d(unif2d_x[r],unif2d_y[r],[max_deg, max_deg])

# For the visualization we use a finer grid (# quadrature points) in the cell
# This means that the Vandermonde interpolation matrix is rectangular

V_rect = {}
for r in range(r_max):
    max_deg = (r+1)
    V_rect[r] = np.polynomial.legendre.legvander2d(unif2d_visual_x,unif2d_visual_y,[max_deg, max_deg])

# Values and grads of basis functions in internal quadrature points, i.e.
# phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
# the gradient has two components due to the x and y derivatives. Repeat for each degree

phi_val_cell={}
for r in range(r_max):
    max_deg = (r+1)
    phi_val_cell[r] = np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[max_deg, max_deg])
    print(phi_val_cell[r])

phi_grad_cell_x={}
phi_grad_cell_y={}
for r in range(r_max):
    max_deg = r+1
    num_coeff = max_deg+1
    phi_grad_cell_x[r]=np.zeros((n_qp,num_coeff*num_coeff))
    phi_grad_cell_y[r]=np.zeros((n_qp,num_coeff*num_coeff))
    temp_vander_x = np.polynomial.legendre.legvander(pts2d_x,max_deg)
    temp_vander_y = np.polynomial.legendre.legvander(pts2d_y,max_deg)
    dLm_x         = np.zeros((n_qp,num_coeff))
    dLm_y         = np.zeros((n_qp,num_coeff))
    coeff = np.zeros(num_coeff)
    for m in range(num_coeff):
        coeff[m]=1.0
        dLm_x[:,m] = np.polynomial.legendre.legval(pts2d_x,np.polynomial.legendre.legder(coeff))
        dLm_y[:,m] = np.polynomial.legendre.legval(pts2d_y,np.polynomial.legendre.legder(coeff)) 
        coeff[m]=0.0

    for m in range(num_coeff):
        for n in range(num_coeff):
            phi_grad_cell_x[r][:,m*num_coeff+n]=np.multiply(temp_vander_x[:,m],dLm_y[:,n])
            phi_grad_cell_y[r][:,m*num_coeff+n]=np.multiply(temp_vander_y[:,n],dLm_x[:,m])

    print(phi_grad_cell_x[r])
