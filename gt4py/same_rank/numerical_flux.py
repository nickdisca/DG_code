import gt4py.gtscript as gtscript
import gt4py as gt
from matmul.matmul_2_4_T import matmul_2_4_T
import numpy as np

from gt4py_config import dtype, backend, backend_opts

@gtscript.function
def flux_function(u_bd):
    fx_0 = u_bd[0,0,0][0]
    fx_1 = u_bd[0,0,0][1]
    fy_0 = u_bd[0,0,0][0]
    fy_1 = u_bd[0,0,0][1]
    return fx_0, fx_1, fy_0, fy_1

def flux_bd_gt(u, fx, fy):
    flux_bd_stencil(u, fx, fy)

@gtscript.stencil(backend=backend, **backend_opts)
def flux_bd_stencil(
    u: gtscript.Field[(dtype, (2,))],
    fx: gtscript.Field[(dtype, (2,))],
    fy: gtscript.Field[(dtype, (2,))]
):
    with computation(PARALLEL), interval(...):
        fx_0, fx_1, fy_0, fy_1, = flux_function(u)
        fx[0,0,0][0] = fx_0
        fx[0,0,0][1] = fx_1

        fy[0,0,0][0] = fy_0
        fy[0,0,0][1] = fy_1

@gtscript.function
def compute_flux_N(fy_n, fy_s, u_n, u_s):
    flux_1 = 0.5 * (fy_n[0,0,0][0] + fy_s[0,+1,0][0]) - 0.5 * (u_s[0,+1,0][0] - u_n[0,0,0][0])
    flux_2 = 0.5 * (fy_n[0,0,0][1] + fy_s[0,+1,0][1]) - 0.5 * (u_s[0,+1,0][1] - u_n[0,0,0][1])
    return flux_1, flux_2

@gtscript.function
def compute_flux_S(fy_s, fy_n, u_s, u_n):
    flux_1 = -0.5 * (fy_s[0,0,0][0] + fy_n[0,-1,0][0]) - 0.5 * (u_n[0,-1,0][0] - u_s[0,0,0][0])
    flux_2 = -0.5 * (fy_s[0,0,0][1] + fy_n[0,-1,0][1]) - 0.5 * (u_n[0,-1,0][1] - u_s[0,0,0][1])
    return flux_1, flux_2

@gtscript.function
def compute_flux_E(fx_e, fx_w, u_e, u_w):
    flux_1 = 0.5 * (fx_e[0,0,0][0] + fx_w[+1,0,0][0]) - 0.5 * (u_w[+1,0,0][0] - u_e[0,0,0][0])
    flux_2 = 0.5 * (fx_e[0,0,0][1] + fx_w[+1,0,0][1]) - 0.5 * (u_w[+1,0,0][1] - u_e[0,0,0][1])
    return flux_1, flux_2

@gtscript.function
def compute_flux_W(fx_w, fx_e, u_w, u_e):
    flux_1 = -0.5 * (fx_w[0,0,0][0] + fx_e[-1,0,0][0]) - 0.5 * (u_e[-1,0,0][0] - u_w[0,0,0][0])
    flux_2 = -0.5 * (fx_w[0,0,0][1] + fx_e[-1,0,0][1]) - 0.5 * (u_e[-1,0,0][1] - u_w[0,0,0][1])
    return flux_1, flux_2

@gtscript.function
def compute_fluxes(f_n, f_s, f_e, f_w, u_n, u_s, u_e, u_w):
    flux_n_0, flux_n_1 = compute_flux_N(f_n, f_s, u_n, u_s)
    flux_s_0, flux_s_1 = compute_flux_S(f_s, f_n, u_s, u_n)
    flux_e_0, flux_e_1 = compute_flux_E(f_e, f_w, u_e, u_w)
    flux_w_0, flux_w_1 = compute_flux_W(f_w, f_e, u_w, u_e)
    return flux_n_0, flux_n_1, flux_s_0, flux_s_1, flux_e_0, flux_e_1, flux_w_0, flux_w_1


@gtscript.stencil(backend=backend, **backend_opts)
def compute_flux_stencil(
    u_n: gtscript.Field[(dtype, (2,))],
    u_s: gtscript.Field[(dtype, (2,))],
    u_e: gtscript.Field[(dtype, (2,))],
    u_w: gtscript.Field[(dtype, (2,))],
    f_n: gtscript.Field[(dtype, (2,))],
    f_s: gtscript.Field[(dtype, (2,))],
    f_e: gtscript.Field[(dtype, (2,))],
    f_w: gtscript.Field[(dtype, (2,))],
    flux_n: gtscript.Field[(dtype, (2,))],
    flux_s: gtscript.Field[(dtype, (2,))],
    flux_w: gtscript.Field[(dtype, (2,))],
    flux_e: gtscript.Field[(dtype, (2,))]
):
    with computation(PARALLEL), interval(...):
        flux_n_0, flux_n_1, flux_s_0, flux_s_1, flux_e_0, flux_e_1, flux_w_0, flux_w_1 = compute_fluxes(f_n, f_s, f_e, f_w, u_n, u_s, u_e, u_w)
        flux_n[0,0,0][0] = flux_n_0
        flux_n[0,0,0][1] = flux_n_1
        flux_s[0,0,0][0] = flux_s_0
        flux_s[0,0,0][1] = flux_s_1
        flux_e[0,0,0][0] = flux_e_0
        flux_e[0,0,0][1] = flux_e_1
        flux_w[0,0,0][0] = flux_w_0
        flux_w[0,0,0][1] = flux_w_1




def compute_flux_gt(u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w):
    nx, ny, nz, vec = u_n.shape
    origins = {
        "u_n": (1,1,0), "u_s": (1,1,0), "u_e": (1,1,0), "u_w": (1,1,0), "f_n": (1,1,0), "f_s": (1,1,0), "f_e": (1,1,0), "f_w": (1,1,0), "flux_n": (0,0,0), "flux_s": (0,0,0), "flux_e": (0,0,0), "flux_w": (0,0,0)
        }
    # origins = {"_all_": (0,0,0)}
    compute_flux_stencil(f_n, f_s, f_e, f_w, u_n, u_s, u_e, u_w, flux_n, flux_s, flux_e, flux_w, origin=origins, domain=(nx-2,ny-2,1))

@gtscript.stencil(backend=backend, **backend_opts)
def integrate_numerical_flux_stencil(
    w: gtscript.Field[(dtype, (2,))],
    f_n: gtscript.Field[(dtype, (2,))],
    f_s: gtscript.Field[(dtype, (2,))],
    f_e: gtscript.Field[(dtype, (2,))],
    f_w: gtscript.Field[(dtype, (2,))],
    phi_n: gtscript.Field[(dtype, (2,4))],
    phi_s: gtscript.Field[(dtype, (2,4))],
    phi_e: gtscript.Field[(dtype, (2,4))],
    phi_w: gtscript.Field[(dtype, (2,4))],
    rhs: gtscript.Field[(dtype, (4,))],
    bd_det_x: float,
    bd_det_y: float

):
    with computation(PARALLEL), interval(...):
        f_n[0,0,0][0] = f_n[0,0,0][0] * w[0,0,0][0]
        f_n[0,0,0][1] = f_n[0,0,0][1] * w[0,0,0][1]
        n_0, n_1, n_2, n_3 = matmul_2_4_T(phi_n, f_n)

        f_s[0,0,0][0] = f_s[0,0,0][0] * w[0,0,0][0]
        f_s[0,0,0][1] = f_s[0,0,0][1] * w[0,0,0][1]
        s_0, s_1, s_2, s_3 = matmul_2_4_T(phi_s, f_s)

        f_e[0,0,0][0] = f_e[0,0,0][0] * w[0,0,0][0]
        f_e[0,0,0][1] = f_e[0,0,0][1] * w[0,0,0][1]
        e_0, e_1, e_2, e_3 = matmul_2_4_T(phi_e, f_e)

        f_w[0,0,0][0] = f_w[0,0,0][0] * w[0,0,0][0]
        f_w[0,0,0][1] = f_w[0,0,0][1] * w[0,0,0][1]
        w_0, w_1, w_2, w_3 = matmul_2_4_T(phi_w, f_w)

        rhs[0,0,0][0] -= bd_det_x * (n_0+s_0) + bd_det_y * (e_0+w_0)
        rhs[0,0,0][1] -= bd_det_x * (n_1+s_1) + bd_det_y * (e_1+w_1)
        rhs[0,0,0][2] -= bd_det_x * (n_2+s_2) + bd_det_y * (e_2+w_2)
        rhs[0,0,0][3] -= bd_det_x * (n_3+s_3) + bd_det_y * (e_3+w_3)
