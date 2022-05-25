import gt4py.gtscript as gtscript
import gt4py as gt
from sympy import O

from gt4py_config import dtype, backend, backend_opts, dim, n_qp, n_qp_1D

@gtscript.stencil(backend=backend, **backend_opts)
def modal2nodal(
    phi: gtscript.Field[(dtype, (dim, dim))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    u_nodal: gtscript.Field[(dtype, (dim,))],
):
    with computation(PARALLEL), interval(...):
        u_nodal = phi @ u_modal

@gtscript.stencil(backend=backend, **backend_opts)
def flux_stencil_swe(
    phi: gtscript.Field[(dtype, (n_qp, dim))],

    h: gtscript.Field[(dtype, (dim,))],
    hu: gtscript.Field[(dtype, (dim,))],
    hv: gtscript.Field[(dtype, (dim,))],

    h_qp: gtscript.Field[(dtype, (n_qp,))],
    hu_qp: gtscript.Field[(dtype, (n_qp,))],
    hv_qp: gtscript.Field[(dtype, (n_qp,))],

    fh_x: gtscript.Field[(dtype, (n_qp,))],
    fh_y: gtscript.Field[(dtype, (n_qp,))],
    fhu_x: gtscript.Field[(dtype, (n_qp,))],
    fhu_y: gtscript.Field[(dtype, (n_qp,))],
    fhv_x: gtscript.Field[(dtype, (n_qp,))],
    fhv_y: gtscript.Field[(dtype, (n_qp,))],

    phi_grad_x: gtscript.Field[(dtype, (n_qp, dim))],
    phi_grad_y: gtscript.Field[(dtype, (n_qp, dim))],
    w: gtscript.Field[(dtype, (n_qp,))],

    rhs_h: gtscript.Field[(dtype, (dim,))],
    rhs_hu: gtscript.Field[(dtype, (dim,))],
    rhs_hv: gtscript.Field[(dtype, (dim,))],
    
    g: float,
    determ: float,
    bd_det_x: float,
    bd_det_y: float
):
    with computation(PARALLEL), interval(...):
        h_qp = phi @ h
        hu_qp = phi @ hu
        hv_qp = phi @ hv

        fh_x = hu_qp
        fh_y = hv_qp

        # BUG: g / 2 * h_qp * h_qp fails
        fhu_x = hu_qp * hu_qp / h_qp + h_qp * h_qp * g / 2
        fhu_y = hu_qp * hv_qp / h_qp

        fhv_x = hu_qp * hv_qp / h_qp
        fhv_y = hv_qp * hv_qp / h_qp + h_qp * h_qp * g / 2

        rhs_h = (phi_grad_x.T @ fh_x / bd_det_x + phi_grad_y.T @ fh_y / bd_det_y) * determ
        rhs_hu = (phi_grad_x.T @ fhu_x / bd_det_x + phi_grad_y.T @ fhu_y / bd_det_y) * determ
        rhs_hv = (phi_grad_x.T @ fhv_x / bd_det_x + phi_grad_y.T @ fhv_y / bd_det_y) * determ

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
def flux_bd_stencil_swe(
    h_n: gtscript.Field[(dtype, (n_qp_1D,))],
    h_s: gtscript.Field[(dtype, (n_qp_1D,))],
    h_e: gtscript.Field[(dtype, (n_qp_1D,))],
    h_w: gtscript.Field[(dtype, (n_qp_1D,))],

    hu_n: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_s: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_e: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_w: gtscript.Field[(dtype, (n_qp_1D,))],

    hv_n: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_s: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_e: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_w: gtscript.Field[(dtype, (n_qp_1D,))],

    f_n_h: gtscript.Field[(dtype, (n_qp_1D,))],
    f_s_h: gtscript.Field[(dtype, (n_qp_1D,))],
    f_e_h: gtscript.Field[(dtype, (n_qp_1D,))],
    f_w_h: gtscript.Field[(dtype, (n_qp_1D,))],

    f_n_hu: gtscript.Field[(dtype, (n_qp_1D,))],
    f_s_hu: gtscript.Field[(dtype, (n_qp_1D,))],
    f_e_hu: gtscript.Field[(dtype, (n_qp_1D,))],
    f_w_hu: gtscript.Field[(dtype, (n_qp_1D,))],

    f_n_hv: gtscript.Field[(dtype, (n_qp_1D,))],
    f_s_hv: gtscript.Field[(dtype, (n_qp_1D,))],
    f_e_hv: gtscript.Field[(dtype, (n_qp_1D,))],
    f_w_hv: gtscript.Field[(dtype, (n_qp_1D,))],

    g: float
):
    with computation(PARALLEL), interval(...):
        f_n_h = hv_n
        f_s_h = hv_s
        f_e_h = hu_e
        f_w_h = hu_w

        f_n_hu = hu_n * hv_n / h_n
        f_s_hu = hu_s * hv_s / h_s
        f_e_hu = hu_e * hu_e / h_e + h_e * h_e * g / 2
        f_w_hu = hu_w * hu_w / h_w + h_w * h_w * g / 2

        f_e_hv = hu_e * hv_e / h_e
        f_w_hv = hu_w * hv_w / h_w
        f_n_hv = hv_n * hv_n / h_n + h_n * h_n * g / 2
        f_s_hv = hv_s * hv_s / h_s + h_s * h_s * g / 2

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
        # BUG: alpha needs to be at end
        flux_n = 0.5 * (f_n + f_s[0,+1,0]) - (u_s[0,+1,0] - u_n) * 0.5 * alpha
        flux_s = -0.5 * (f_s + f_n[0,-1,0]) - (u_n[0,-1,0] - u_s) * 0.5 * alpha
        flux_e = 0.5 * (f_e + f_w[+1,0,0]) - (u_w[+1,0,0] - u_e) * 0.5 * alpha
        flux_w = -0.5 * (f_w + f_e[-1,0,0]) -  (u_e[-1,0,0] - u_w) * 0.5 * alpha

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

    radius: float,
    bd_det_x: float,
    bd_det_y: float
):
    with computation(PARALLEL), interval(...):
        flux_n *= w
        flux_s *= w
        flux_e *= w
        flux_w *= w

        rhs -= (phi_bd_N.T @ flux_n) * bd_det_x
        rhs -= (phi_bd_S.T @ flux_s) * bd_det_x
        rhs -= (phi_bd_E.T @ flux_e) * bd_det_y
        rhs -= (phi_bd_W.T @ flux_w) * bd_det_y

        rhs = (inv_mass @ rhs) / radius
        
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
    rhs: gtscript.Field[(dtype, (dim,))],
    k2: gtscript.Field[(dtype, (dim,))],
    u_modal: gtscript.Field[(dtype, (dim,))],
    dt: float,
    out: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out = u_modal + dt / 2 * (rhs + k2)

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
