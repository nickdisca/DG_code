import numpy as np
import gt4py.gtscript as gtscript
import gt4py as gt
from flux_function import flux_function_gt, integrate_flux_stencil
from modal_conversion import modal2qp_gt, modal2bd_gt
from numerical_flux import flux_bd_gt, compute_flux_stencil

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

def integrate_flux(w, fx, fy, phi_grad_x, phi_grad_y, determ, bd_det_x, bd_det_y):
    nx, ny, nz, vec = fx.shape
    out = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    integrate_flux_stencil(w, fx, fy, phi_grad_x, phi_grad_y, out, determ, bd_det_x, bd_det_y)
    return out


def compute_rhs(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp, hx, hy):
    determ = hx * hy / 4
    bd_det_x = hx / 2
    bd_det_y = hy / 2

    # --- Flux Integrals ---
    u_qp = modal2qp_gt(vander.phi_gt, uM_gt)
    fx, fy = flux_function_gt(u_qp)
    rhs = integrate_flux(wts2d, fx, fy, vander.grad_phi_x_gt, vander.grad_phi_y_gt, determ, bd_det_x, bd_det_y)

    # --- Boundary Numerical Flux ---
    u_n = modal2bd_gt(vander.phi_bd_N_gt, uM_gt)
    u_s = modal2bd_gt(vander.phi_bd_S_gt, uM_gt)
    u_e = modal2bd_gt(vander.phi_bd_E_gt, uM_gt)
    u_w = modal2bd_gt(vander.phi_bd_W_gt, uM_gt)

    _, f_n = flux_bd_gt(u_n)
    _, f_s = flux_bd_gt(u_s)
    f_e, _ = flux_bd_gt(u_e)
    f_w, _ = flux_bd_gt(u_w)

    compute_flux_stencil(u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w)

    return rhs





