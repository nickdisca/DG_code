import numpy as np
import gt4py.gtscript as gtscript
import gt4py as gt
from flux_function import flux_function_gt, integrate_flux_stencil
from modal_conversion import modal2qp_gt, modal2bd_gt
from numerical_flux import flux_bd_gt, compute_flux_gt, integrate_numerical_flux_stencil

from matmul.matmul_4_4 import matmul_4_4

dtype = np.float64
backend = "gtc:numpy"
backend_opts = {
    "rebuild": True
}

# @gtscript.stencil(backend=backend, **backend_opts)
# def elemwise_mult(
#     a: gtscript.Field[(dtype, (4,))],
#     b: gtscript.Field[(dtype, (4,))],
#     out: gtscript.Field[(dtype, (4,))]
# ):
#     with computation(PARALLEL), interval(...):
#         out[0,0,0][0] = a[0,0,0][0] * b[0,0,0][0]
#         out[0,0,0][1] = a[0,0,0][1] * b[0,0,0][1]
#         out[0,0,0][2] = a[0,0,0][2] * b[0,0,0][2]
#         out[0,0,0][3] = a[0,0,0][3] * b[0,0,0][3]

@gtscript.stencil(backend=backend, **backend_opts)
def inv_mass_stencil(
    inv_mass: gtscript.Field[(dtype, (4,4))],
    rhs: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        a_0, a_1, a_2, a_3 = matmul_4_4(inv_mass, rhs)
        rhs[0,0,0][0] = a_0
        rhs[0,0,0][1] = a_1
        rhs[0,0,0][2] = a_2
        rhs[0,0,0][3] = a_3


def integrate_flux(w, fx, fy, vander, determ, bd_det_x, bd_det_y):
    nx, ny, nz, vec = fx.shape
    phi_grad_x = vander.grad_phi_x_gt
    phi_grad_y = vander.grad_phi_y_gt
    out = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    integrate_flux_stencil(w, fx, fy, phi_grad_x, phi_grad_y, out, determ, bd_det_x, bd_det_y)
    return out

def integrate_numerical_flux(w, f_n, f_s, f_e, f_w, vander, bd_det_x, bd_det_y):
    nx, ny, nz, vec = f_n.shape
    phi_n = vander.phi_bd_N_gt
    phi_s = vander.phi_bd_S_gt
    phi_e = vander.phi_bd_E_gt
    phi_w = vander.phi_bd_W_gt
    out = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (4,)))
    integrate_numerical_flux_stencil(w, f_n, f_s, f_e, f_w, phi_n, phi_s, phi_e, phi_w, out, bd_det_x, bd_det_y)
    return out

@gtscript.stencil(backend=backend, **backend_opts)
def subtract_boundary_term_stencil(
    rhs: gtscript.Field[(dtype, (4,))],
    boundary_term: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        rhs[0,0,0][0] -= boundary_term[0,0,0][0]
        rhs[0,0,0][1] -= boundary_term[0,0,0][1]
        rhs[0,0,0][2] -= boundary_term[0,0,0][2]
        rhs[0,0,0][3] -= boundary_term[0,0,0][3]

@gtscript.stencil(backend=backend, **backend_opts)
def runge_kuta_stencil(
    u_modal: gtscript.Field[(dtype, (4,))],
    rhs: gtscript.Field[(dtype, (4,))],
    dt: float
):
    with computation(PARALLEL), interval(...):
        u_modal[0,0,0][0] += dt * rhs[0,0,0][0]
        u_modal[0,0,0][1] += dt * rhs[0,0,0][1]
        u_modal[0,0,0][2] += dt * rhs[0,0,0][2]
        u_modal[0,0,0][3] += dt * rhs[0,0,0][3]

def compute_rhs(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp, hx, hy, nx, ny, dt):
    determ = hx * hy / 4
    bd_det_x = hx / 2
    bd_det_y = hy / 2
    radius = 1

    # --- Flux Integrals ---
    u_qp = modal2qp_gt(vander.phi_gt, uM_gt)
    fx, fy = flux_function_gt(u_qp)
    rhs = integrate_flux(wts2d, fx, fy, vander, determ, bd_det_x, bd_det_y)

    # --- Boundary Numerical Flux ---
    u_n = modal2bd_gt(vander.phi_bd_N_gt, uM_gt)
    u_s = modal2bd_gt(vander.phi_bd_S_gt, uM_gt)
    u_e = modal2bd_gt(vander.phi_bd_E_gt, uM_gt)
    u_w = modal2bd_gt(vander.phi_bd_W_gt, uM_gt)

    _, f_n = flux_bd_gt(u_n)
    _, f_s = flux_bd_gt(u_s)
    f_e, _ = flux_bd_gt(u_e)
    f_w, _ = flux_bd_gt(u_w)

    flux_n, flux_s, flux_e, flux_w = compute_flux_gt(u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w)
    boundary_term = integrate_numerical_flux(wts1d, flux_n, flux_s, flux_w, flux_e, vander, bd_det_x, bd_det_y)
    subtract_boundary_term_stencil(rhs, boundary_term)

    inv_mass_stencil(inv_mass, rhs) 
    runge_kuta_stencil(uM_gt, rhs, dt)
    return rhs





