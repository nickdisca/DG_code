import gt4py.gtscript as gtscript
import gt4py as gt

from gt4py_config import dtype, backend, backend_opts, dim, n_qp, n_qp_1D

import numpy as np

@gtscript.stencil(backend=backend, **backend_opts)
def modal2nodal(
    phi: gtscript.Field[(dtype, (dim, dim))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    u_nodal: gtscript.Field[(dtype, (dim,))],
):
    with computation(PARALLEL), interval(...):
        u_nodal = phi @ u_modal

@gtscript.stencil(backend=backend, **backend_opts)
def flux_stencil(
    phi: gtscript.Field[(dtype, (n_qp, dim))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    u_qp: gtscript.Field[(dtype, (n_qp,))],

    # fx: gtscript.Field[(dtype, (n_qp,))],
    # fy: gtscript.Field[(dtype, (n_qp,))],

    phi_grad_x: gtscript.Field[(dtype, (n_qp, dim))],
    phi_grad_y: gtscript.Field[(dtype, (n_qp, dim))],
    w: gtscript.Field[(dtype, (n_qp,))],
    rhs: gtscript.Field[(dtype, (dim,))],
    determ: float,
    bd_det_x: float,
    bd_det_y: float
):
    tmp: gtscript.Field[(np.float64, (4,))] = 0
    with computation(PARALLEL), interval(...):
        tmp = phi @ u_modal

        # u_qp = phi @ u_modal
        # fx = tmp * w
        # fy = tmp * w
        # rhs = (phi_grad_x.T @ fx / bd_det_x + phi_grad_y.T @ fy / bd_det_y) * determ
        # pass

@gtscript.stencil(backend=backend, **backend_opts)
def modal2bd(
    phi_bd_N: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_S: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_E: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_W: gtscript.Field[(dtype, (n_qp_1D, dim))],

    u_n: gtscript.Field[(dtype, (n_qp_1D,))],
    u_s: gtscript.Field[(dtype, (n_qp_1D,))],
    u_e: gtscript.Field[(dtype, (n_qp_1D,))],
    u_w: gtscript.Field[(dtype, (n_qp_1D,))],

    u_modal: gtscript.Field[(dtype, (dim,))],
    
):
    with computation(PARALLEL), interval(...):
        u_n = phi_bd_N @ u_modal
        u_s = phi_bd_S @ u_modal
        u_e = phi_bd_E @ u_modal
        u_w = phi_bd_W @ u_modal



@gtscript.stencil(backend=backend, **backend_opts)
def flux_bd_stencil(
    u_n: gtscript.Field[(dtype, (n_qp_1D,))],
    u_s: gtscript.Field[(dtype, (n_qp_1D,))],
    u_e: gtscript.Field[(dtype, (n_qp_1D,))],
    u_w: gtscript.Field[(dtype, (n_qp_1D,))],

    f_n: gtscript.Field[(dtype, (n_qp_1D,))],
    f_s: gtscript.Field[(dtype, (n_qp_1D,))],
    f_e: gtscript.Field[(dtype, (n_qp_1D,))],
    f_w: gtscript.Field[(dtype, (n_qp_1D,))]
):
    with computation(PARALLEL), interval(...):
        f_n = u_n
        f_s = u_s
        f_e = u_e
        f_w = u_w

@gtscript.stencil(backend=backend, **backend_opts)
def compute_num_flux(
    u_n: gtscript.Field[(dtype, (n_qp_1D,))],
    u_s: gtscript.Field[(dtype, (n_qp_1D,))],
    u_e: gtscript.Field[(dtype, (n_qp_1D,))],
    u_w: gtscript.Field[(dtype, (n_qp_1D,))],

    f_n: gtscript.Field[(dtype, (n_qp_1D,))],
    f_s: gtscript.Field[(dtype, (n_qp_1D,))],
    f_e: gtscript.Field[(dtype, (n_qp_1D,))],
    f_w: gtscript.Field[(dtype, (n_qp_1D,))],

    flux_n: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_s: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_e: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_w: gtscript.Field[(dtype, (n_qp_1D,))],

    alpha: float

):
    with computation(PARALLEL), interval(...):
        # flux_n = 0.5 * (f_n + f_s[0,+1,0]) - 0.5 * (u_s[0,+1,0] - u_n)
        # flux_s = -0.5 * (f_s + f_n[0,-1,0]) - 0.5 * (u_n[0,-1,0] - u_s)
        # flux_e = 0.5 * (f_e + f_w[+1,0,0]) - 0.5 * (u_w[+1,0,0] - u_e)
        # flux_w = -0.5 * (f_w + f_e[-1,0,0]) - 0.5 * (u_e[-1,0,0] - u_w)

        flux_n = 0.5 * (f_n + f_s[0,+1,0]) - 0.5 * alpha * (u_s[0,+1,0] - u_n)
        flux_s = -0.5 * (f_s + f_n[0,-1,0]) - 0.5 * alpha * (u_n[0,-1,0] - u_s)
        flux_e = 0.5 * (f_e + f_w[+1,0,0]) - 0.5 * alpha * (u_w[+1,0,0] - u_e)
        flux_w = -0.5 * (f_w + f_e[-1,0,0]) - 0.5 * alpha * (u_e[-1,0,0] - u_w)

@gtscript.stencil(backend=backend, **backend_opts)
def integrate_num_flux(
    phi_bd_N: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_S: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_E: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_W: gtscript.Field[(dtype, (n_qp_1D, dim))],

    flux_n: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_s: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_e: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_w: gtscript.Field[(dtype, (n_qp_1D,))],

    w: gtscript.Field[(dtype, (n_qp_1D,))],

    rhs: gtscript.Field[(dtype, (dim,))],
    inv_mass: gtscript.Field[(dtype, (dim, dim))],

    bd_det_x: float,
    bd_det_y: float
):
    with computation(PARALLEL), interval(...):
        flux_n *= w
        flux_s *= w
        flux_e *= w
        flux_w *= w

        rhs -= phi_bd_N.T @ flux_n * bd_det_x
        rhs -= phi_bd_S.T @ flux_s * bd_det_x
        rhs -= phi_bd_E.T @ flux_e * bd_det_y
        rhs -= phi_bd_W.T @ flux_w * bd_det_y

        rhs = (inv_mass @ rhs)
        
@gtscript.stencil(backend=backend, **backend_opts)
def rk_step1(
    rhs: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + dt * rhs

@gtscript.stencil(backend=backend, **backend_opts)
def rk_step2(
    k1: gtscript.Field[(dtype, (dim,))],
    k2: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + dt / 2 * (k1 + k2)

@gtscript.stencil(backend=backend, **backend_opts)
def rk_step2_3(
    k1: gtscript.Field[(dtype, (dim,))],
    k2: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + 0.25 * dt * (k1 + k2)

@gtscript.stencil(backend=backend, **backend_opts)
def rk_step3_3(
    k1: gtscript.Field[(dtype, (dim,))],
    k2: gtscript.Field[(dtype, (dim,))],
    k3: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + dt / 6 * (k1 + k2 + 4 * k3)

@gtscript.stencil(backend=backend, **backend_opts)
def rk_step1_4(
    k1: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + dt / 2 * k1

@gtscript.stencil(backend=backend, **backend_opts)
def rk_step2_4(
    k1: gtscript.Field[(dtype, (dim,))],
    k2: gtscript.Field[(dtype, (dim,))],
    k3: gtscript.Field[(dtype, (dim,))],
    k4: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
