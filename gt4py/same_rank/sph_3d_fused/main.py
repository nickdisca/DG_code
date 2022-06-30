# %%
import numpy as np
import time
import gt4py as gt
import quadpy as qp

from vander import Vander
from initial_conditions import set_initial_conditions
from modal_conversion import nodal2modal_gt, modal2nodal_gt, integration
from compute_mass import compute_mass
from run import run
from plotter import Plotter
from gt4py_config import backend, dtype, r, n_qp_1D, runge_kutta, nx, nz

import plotly
from scalene import scalene_profiler

from stencils import modal2nodal
# silence warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

debug = False

# %%
# Radius of the earth (for spherical geometry)


# Equation type
eq_type="swe_sphere"

# domain
if eq_type == 'linear':
    a = 0; b = 1; c = 0; d =1
    radius=1
elif eq_type == 'swe':
    a = 0; b = 1e7; c = 0; d = 1e7
    radius=1
elif eq_type == 'swe_sphere':
    a = 0; b = 2*np.pi; c = -np.pi/2; d = np.pi/2
    radius=6.37122e6

# number of elements in X and Y
ny = nx;

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
# courant = 0.1
# dt = courant * dx / (r + 1)

if eq_type == 'linear':
    T = 1
elif eq_type == 'swe':
    T = 1e7
elif eq_type == 'swe_sphere':
    day_in_sec = 86400
    T = 1 * day_in_sec
    # alpha = 170.0 
    dt = 5.0

# niter = int(T / dt)
# niter = 10000

# plotting
# plot_freq = int(niter / 10)
plot_type = "contour"

# plot_freq = 100
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
vander = Vander(nx, ny, nz, dim, r, n_qp, pts2d_x, pts2d_y, pts, wts2d, backend=backend)
vander_end = time.perf_counter()

neq, u0_nodal, coriolis = set_initial_conditions(x_c, y_c, a, b, c, d, radius, dim, n_qp, vander, pts2d_x, pts2d_y, eq_type)

if eq_type == 'swe' or eq_type == "swe_sphere":
    h0, u0, v0 = u0_nodal

    # identical systems in z component
    if nz > 1:
        h0 = np.repeat(h0, nz, axis=2)
        u0 = np.repeat(u0, nz, axis=2)
        v0 = np.repeat(v0, nz, axis=2)
        coriolis = np.repeat(coriolis, nz, axis=2)

    g = 9.80616
    alpha = np.max(np.sqrt(g*h0) + np.sqrt(u0**2 + v0**2))
    courant = 0.009
    dt = courant * min((radius * np.sin(hx) * np.sin(hy), radius * np.sin(hx) * np.cos(hy))) / ((r+1) * alpha)
    # niter = int(T/dt)
    niter = 100000
    plot_freq = 10000

    h0_nodal_gt = gt.storage.from_array(data=h0,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (dim,)))
    hu0_nodal_gt = gt.storage.from_array(data=u0*h0,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (dim,)))
    hv0_nodal_gt = gt.storage.from_array(data=v0*h0,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (dim,)))
    coriolis_gt = gt.storage.from_array(data=coriolis,
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (n_qp,)))

plotter = Plotter(x_c, y_c, r+1, nx, ny, neq, hx, hy, plot_freq, plot_type)

# if not debug:
#     plotter.plot_solution(u0_nodal_gt, init=True, plot_type=plotter.plot_type)
# plotter.plot_solution((h0_nodal_gt, hu0_nodal_gt, hv0_nodal_gt), init=True, title=f'INIT: {nx = }; {nz = } | {r = }; {runge_kutta = } | {backend = }', plot_type=plotter.plot_type, fname=f'init_{backend}.png')
# time.sleep(20)
h0_modal_gt = gt.storage.zeros(
        backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (dim,))
)
hu0_modal_gt = gt.storage.zeros(
    backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (dim,))
)
hv0_modal_gt = gt.storage.zeros(
    backend=backend, default_origin=(0,0,0), shape=(nx,ny,nz), dtype=(dtype, (dim,))
)

modal2nodal(vander.inv_vander_gt, h0_nodal_gt, h0_modal_gt)
modal2nodal(vander.inv_vander_gt, hu0_nodal_gt, hu0_modal_gt)
modal2nodal(vander.inv_vander_gt, hv0_nodal_gt, hv0_modal_gt)

plotter.plot_solution((h0_nodal_gt, hu0_nodal_gt, hv0_nodal_gt), init=True, plot_type=plotter.plot_type, fname='init.png')

mass, inv_mass, cos_factor, sin_factor, cos_n, cos_s = compute_mass(vander.phi_val_cell, wts2d, nx, ny, r, hx, hy, y_c, pts2d_y, pts, eq_type)

inv_mass_gt = gt.storage.from_array(inv_mass, backend=backend, default_origin=(0,), shape=(ny,), dtype=(dtype, (dim, dim)), mask=[False, True, False])
wts2d_gt = gt.storage.from_array(wts2d, backend=backend, default_origin=(0,), shape=(nz,), dtype=(dtype, (n_qp,)), mask=[False, False, True])
wts1d_gt = gt.storage.from_array(wts, backend=backend, default_origin=(0,), shape=(nz,), dtype=(dtype, (n_qp_1D,)), mask=[False, False, True])
cos_gt = gt.storage.from_array(cos_factor, backend=backend, default_origin=(0,), shape=(ny,), dtype=(dtype, (n_qp,)), mask=[False, True, False])
sin_gt = gt.storage.from_array(sin_factor, backend=backend, default_origin=(0,), shape=(ny,), dtype=(dtype, (n_qp,)), mask=[False, True, False])
cos_n_gt = gt.storage.from_array(cos_n, backend=backend, default_origin=(0,), shape=(ny,), dtype=(dtype, (n_qp_1D,)), mask=[False, True, False])
cos_s_gt = gt.storage.from_array(cos_s, backend=backend, default_origin=(0,), shape=(ny,), dtype=(dtype, (n_qp_1D,)), mask=[False, True, False])

print(f'\n\n--- Backend = {backend} ---')
print(f'Domain: {nx = }; {ny = }; {nz = }\nTimesteping: {dt = }; {niter = }')
print(f'Order: space {r+1}; time {runge_kutta}')
print(f'Diffusion constant flux: {alpha = }')

run((h0_modal_gt, hu0_modal_gt, hv0_modal_gt), vander, inv_mass_gt, wts2d_gt, wts1d_gt, dim, n_qp_1D, n_qp, hx, hy, nx, ny, nz, cos_gt, sin_gt, (cos_n_gt, cos_s_gt), coriolis_gt, radius, alpha, dt, niter, plotter)

# Timinig
print(f'Vander: {vander_end - vander_start}s')

# Plot final time
if debug:
    init = True
else:
    init = False

modal2nodal(vander.vander_gt, h0_modal_gt, h0_nodal_gt)
modal2nodal(vander.vander_gt, hu0_modal_gt, hu0_nodal_gt)
modal2nodal(vander.vander_gt, hv0_modal_gt, hv0_nodal_gt)
print(f'{type(hu0_nodal_gt) = }')

# comment
h0_nodal_gt.device_to_host()
hu0_nodal_gt.device_to_host()
hv0_nodal_gt.device_to_host()

h_f = np.asarray(h0_nodal_gt)
hu_f = np.asarray(hu0_nodal_gt)
hv_f = np.asarray(hv0_nodal_gt)
# h0_nodal_gt.synchronize()
# hu0_nodal_gt.synchronize()
# hv0_nodal_gt.synchronize()

# h_f = modal2nodal_gt(vander.vander_gt, h0_modal_gt)
# # hu_f = modal2nodal_gt(vander.vander_gt, hu0_modal_gt)

# hu_f = np.einsum('ijklm,ijkm->ijkl', vander.vander_gt, np.asarray(hu0_modal_gt))

# hv_f = modal2nodal_gt(vander.vander_gt, hv0_modal_gt)

plotter.plot_solution((h_f, hu_f, hv_f), init=init, title=f'{nx = }; {nz = } | {r = }; {runge_kutta = } | {dt = :.1f}; {niter = } | {backend = }', fname=f'nx{nx}_nz{nz}_p{r+1}_rk{runge_kutta}_T{int(dt*niter)}_{backend}.png', show=False)
# plotter.plot_solution((h0_nodal_gt, hu0_nodal_gt, hv0_nodal_gt), init=init, title=f'{nx = }; {nz = } | {r = }; {runge_kutta = } | {dt = :.1f}; {niter = } | {backend = }', fname=f'nx{nx}_nz{nz}_p{r+1}_rk{runge_kutta}_T{int(dt*niter)}_{backend}.png', show=False)
