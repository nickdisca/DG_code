import gt4py.gtscript as gtscript
import gt4py as gt

from gt4py_config import dtype, backend, backend_opts

@gtscript.stencil(backend=backend, **backend_opts)
def flux_stencil(
    phi: gtscript.Field[(dtype, (4, 4))],
    u_modal: gtscript.Field[(dtype, (4,))],
    u_qp: gtscript.Field[(dtype, (4,))],

    fx: gtscript.Field[(dtype, (4,))],
    fy: gtscript.Field[(dtype, (4,))],

    phi_grad_x: gtscript.Field[(dtype, (4, 4))],
    phi_grad_y: gtscript.Field[(dtype, (4, 4))],
    w: gtscript.Field[(dtype, (4,))],
    rhs: gtscript.Field[(dtype, (4,))],
    determ: float,
    bd_det_x: float,
    bd_det_y: float
):
    with computation(PARALLEL), interval(...):
        u_qp = phi @ u_modal
        fx = u_qp * w
        fy = u_qp * w
        rhs = (phi_grad_x.T @ fx / bd_det_x + phi_grad_y.T @ fy / bd_det_y) * determ

@gtscript.stencil(backend=backend, **backend_opts)
def modal2bd(
    phi_bd_N: gtscript.Field[(dtype, (2, 4))],
    phi_bd_S: gtscript.Field[(dtype, (2, 4))],
    phi_bd_E: gtscript.Field[(dtype, (2, 4))],
    phi_bd_W: gtscript.Field[(dtype, (2, 4))],

    u_n: gtscript.Field[(dtype, (2,))],
    u_s: gtscript.Field[(dtype, (2,))],
    u_e: gtscript.Field[(dtype, (2,))],
    u_w: gtscript.Field[(dtype, (2,))],

    u_modal: gtscript.Field[(dtype, (4,))],
    
):
    with computation(PARALLEL), interval(...):
        u_n = phi_bd_N @ u_modal
        u_s = phi_bd_S @ u_modal
        u_e = phi_bd_E @ u_modal
        u_w = phi_bd_W @ u_modal



@gtscript.stencil(backend=backend, **backend_opts)
def flux_bd_stencil(
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
        f_n = u_n
        f_s = u_s
        f_e = u_e
        f_w = u_w

@gtscript.stencil(backend=backend, **backend_opts)
def compute_num_flux(
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
    flux_e: gtscript.Field[(dtype, (2,))],
    flux_w: gtscript.Field[(dtype, (2,))]
):
    with computation(PARALLEL), interval(...):
        flux_n = 0.5 * (f_n + f_s[0,+1,0]) - 0.5 * (u_s[0,+1,0] - u_n)
        flux_s = -0.5 * (f_s + f_n[0,-1,0]) - 0.5 * (u_n[0,-1,0] - u_s)
        flux_e = 0.5 * (f_e + f_w[+1,0,0]) - 0.5 * (u_w[+1,0,0] - u_e)
        flux_w = -0.5 * (f_w + f_e[-1,0,0]) - 0.5 * (u_e[-1,0,0] - u_w)

@gtscript.stencil(backend=backend, **backend_opts)
def integrate_num_flux(
    phi_bd_N: gtscript.Field[(dtype, (2, 4))],
    phi_bd_S: gtscript.Field[(dtype, (2, 4))],
    phi_bd_E: gtscript.Field[(dtype, (2, 4))],
    phi_bd_W: gtscript.Field[(dtype, (2, 4))],

    flux_n: gtscript.Field[(dtype, (2,))],
    flux_s: gtscript.Field[(dtype, (2,))],
    flux_e: gtscript.Field[(dtype, (2,))],
    flux_w: gtscript.Field[(dtype, (2,))],

    w: gtscript.Field[(dtype, (2,))],

    rhs: gtscript.Field[(dtype, (4,))],
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
        
@gtscript.stencil(backend=backend, **backend_opts)
def rk_step(
    inv_mass: gtscript.Field[(dtype, (4,4))],
    rhs: gtscript.Field[(dtype, (4,))],
    u_modal: gtscript.Field[(dtype, (4,))],
    dt: float
):
    with computation(PARALLEL), interval(...):
        # BUG: parantheses around matmul is required !!!
        u_modal += dt * (inv_mass @ rhs)


@gtscript.stencil(backend=backend, **backend_opts)
def modal2nodal(
    phi: gtscript.Field[(dtype, (4,4))],
    u_modal: gtscript.Field[(dtype, (4,))],
    u_nodal: gtscript.Field[(dtype, (4,))],
):
    with computation(PARALLEL), interval(...):
        u_nodal = phi @ u_modal
