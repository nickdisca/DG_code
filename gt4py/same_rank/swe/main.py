# %%
import numpy as np
import time
import gt4py as gt
import gt4py.gtscript as gtscript
import quadpy as qp
from numpy.polynomial import legendre as L

from vander import Vander
from initial_conditions import set_initial_conditions
from modal_conversion import nodal2modal_gt, modal2nodal_gt, integration
from compute_mass import compute_mass
from run import run
from plotter import Plotter
from gt4py_config import backend, dtype, backend_opts, r, n_qp_1D, runge_kutta

import plotly
from scalene import scalene_profiler

# silence warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

debug = False

# %%
# Radius of the earth (for spherical geometry)
# radius=6.37122e6
radius=1

# Equation type
eq_type="swe"

# domain
if eq_type == 'linear':
    a = 0; b = 1; c = 0; d =1
elif eq_type == 'swe':
    a = 0; b = 1e7; c = 0; d = 1e7

# number of elements in X and Y
nx = 50; ny = 50

hx = (b-a)/nx; hy = (d-c)/ny
dx = np.min((hx, hy))

# cardinality
dim=(r+1)**2

# Type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg"

# Number of quadrature points in one dimension
# n_qp_1D=4

# Number of quadrature points
n_qp=n_qp_1D*n_qp_1D


# timestep
# dt = 1e-4
courant = 0.0001

dt = courant * dx / (r + 1)
alpha = courant * dx / dt

if eq_type == 'linear':
    T = 1
elif eq_type == 'swe':
    T = 1e7
niter = int(T / dt)

# plotting
plot_freq = int(niter / 10)
plot_type = "contour"

plot_freq = 100
# %%
# rdist_gt = degree_distribution("unif",nx,ny,r_max);

if debug:
    nx = 2; ny = 2
    niter = 1
    plot_freq = niter + 10

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

# all matrices are the same size but lower orders are padded!
vander_start = time.perf_counter()
vander = Vander(nx, ny, dim, r, n_qp, pts2d_x, pts2d_y, pts, wts2d, backend=backend)
vander_end = time.perf_counter()

neq, u0_nodal = set_initial_conditions(x_c, y_c, a, b, c, d, dim, vander, eq_type)

if eq_type == 'swe':
    h0 = u0_nodal[0]
    u0 = u0_nodal[1]
    v0 = u0_nodal[2]


    h0_nodal_gt = gt.storage.from_array(data=h0,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,1), dtype=(dtype, (dim,)))
    hu0_nodal_gt = gt.storage.from_array(data=u0*h0,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,1), dtype=(dtype, (dim,)))
    hv0_nodal_gt = gt.storage.from_array(data=v0*h0,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,1), dtype=(dtype, (dim,)))

plotter = Plotter(x_c, y_c, r+1, nx, ny, neq, hx, hy, plot_freq, plot_type)

# if not debug:
#     plotter.plot_solution(u0_nodal_gt, init=True, plot_type=plotter.plot_type)
plotter.plot_solution(h0_nodal_gt, init=True, plot_type=plotter.plot_type)

h0_ref = nodal2modal_gt(vander.inv_vander_gt, h0_nodal_gt)
h0_modal_gt = nodal2modal_gt(vander.inv_vander_gt, h0_nodal_gt)
hu0_modal_gt = nodal2modal_gt(vander.inv_vander_gt, hu0_nodal_gt)
hv0_modal_gt = nodal2modal_gt(vander.inv_vander_gt, hv0_nodal_gt)

mass, inv_mass = compute_mass(vander.phi_val_cell, wts2d, nx, ny, r, hx, hy, y_c, pts2d_y, eq_type)

inv_mass_gt = gt.storage.from_array(inv_mass, backend=backend, default_origin=(0,0,0), shape=(nx,ny, 1), dtype=(dtype, (dim, dim)))

wts2d_gt = gt.storage.from_array(wts2d, backend=backend, default_origin=(0,0,0), shape=(nx,ny, 1), dtype=(dtype, (n_qp, )))

wts1d_gt = gt.storage.from_array(wts, backend=backend, default_origin=(0,0,0), shape=(nx,ny, 1), dtype=(dtype, (n_qp_1D, )))

print(f'\n\n--- Backend = {backend} ---')
print(f'Domain: {nx = }; {ny = }\nTimesteping: {dt = }; {niter = }')
print(f'Order: space {r+1}; time {runge_kutta}')

run((h0_modal_gt, hu0_modal_gt, hv0_modal_gt), vander, inv_mass_gt, wts2d_gt, wts1d_gt, dim, n_qp_1D, n_qp, hx, hy, nx, ny, alpha, dt, niter, plotter)

u_final_nodal = modal2nodal_gt(vander.vander_gt, h0_modal_gt)

if backend == "cuda":
    u_final_nodal.device_to_host()

u_final = np.asarray(u_final_nodal)

# Timinig
print(f'Vander: {vander_end - vander_start}s')

# Error
print('--- Error ---')
# l2_error = np.sum(np.sqrt((u0_nodal - u_final)**2) * wts2d)
# l2_error = np.sum(np.sqrt((u0_nodal - u_final)**2) * wts2d) / np.sum(np.sqrt(u0_nodal**2) * wts2d)
determ = hx * hy / 4
tmp = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
    shape=(nx, ny, 1), dtype=(dtype, (n_qp,)))
integration(vander.phi_gt, wts2d_gt, np.sqrt((h0_ref - u_final)**2), determ, tmp)
l2_error = np.sum(tmp)
print(f'{l2_error=}')

# Plot final time
if debug:
    init = True
else:
    init = False
# plotter.plot_solution(u_final_nodal, init=init, show=False, save=True)
