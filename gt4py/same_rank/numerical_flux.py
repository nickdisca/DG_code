import gt4py.gtscript as gtscript
import gt4py as gt
from matmul.matmul_2_4_T import matmul_2_4_T
import numpy as np

dtype = np.float64
backend = "gtc:numpy"
backend_opts = {
    "rebuild": True
}

@gtscript.function
def flux_function(u_bd):
    fx_0 = u_bd[0,0,0][0]
    fx_1 = u_bd[0,0,0][1]
    fy_0 = u_bd[0,0,0][0]
    fy_1 = u_bd[0,0,0][1]
    return fx_0, fx_1, fy_0, fy_1

def flux_bd_gt(u):
    nx, ny, nz, vec = u.shape
    fx = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    fy = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    flux_bd_stencil(u, fx, fy)
    return fx, fy

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
    flux = 1/2 * (fy_n[0,0,0] + fy_s[0,+1,0]) - 1/2 * (u_s[0,+1,0] - u_n[0,0,0])
    return flux

@gtscript.function
def compute_flux_S(fy_s, fy_n, u_s, u_n):
    flux = -1/2 * (fy_s[0,0,0] + fy_n[0,-1,0]) - 1/2 * (u_n[0,-1,0] - u_s[0,0,0])
    return flux

@gtscript.function
def compute_flux_E(fx_e, fx_w, u_e, u_w):
    flux = 1/2 * (fx_e[0,0,0] + fx_w[+1,0,0]) - 1/2 * (u_w[+1,0,0] - u_e[0,0,0])
    return flux

@gtscript.function
def compute_flux_W(fx_w, fx_e, u_w, u_e):
    flux = -1/2 * (fx_w[0,0,0] + fx_e[-1,0,0]) - 1/2 * (u_e[-1,0,0] - u_w[0,0,0])
    return flux

@gtscript.function
def compute_fluxes(f_n, f_s, f_e, f_w, u_n, u_s, u_e, u_w):
    flux_n = compute_flux_N(f_n, f_s, u_n, u_s)
    flux_s = compute_flux_S(f_s, f_n, u_s, u_n)
    flux_e = compute_flux_S(f_e, f_w, u_e, u_w)
    flux_w = compute_flux_S(f_w, f_e, u_w, u_e)
    return flux_n, flux_s, flux_e, flux_w


@gtscript.stencil(backend=backend, **backend_opts)
def compute_flux_stencil(
    u_n: gtscript.Field[(dtype, (2,))],
    u_s: gtscript.Field[(dtype, (2,))],
    u_e: gtscript.Field[(dtype, (2,))],
    u_w: gtscript.Field[(dtype, (2,))],
    f_n: gtscript.Field[(dtype, (2,))],
    f_s: gtscript.Field[(dtype, (2,))],
    f_e: gtscript.Field[(dtype, (2,))],
    f_w: gtscript.Field[(dtype, (2,))]
):
    with computation(PARALLEL), interval(...):
        flux_n, flux_s, flux_e, flux_w, = compute_fluxes(f_n, f_s, f_e, f_w, u_n, u_s, u_e, u_w)