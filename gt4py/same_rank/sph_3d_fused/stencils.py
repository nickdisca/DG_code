import gt4py.gtscript as gtscript
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
def fused_internal_stencils(
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

    cos_fact: gtscript.Field[(dtype, (dim,))], 
    sin_fact: gtscript.Field[(dtype, (dim,))], 
    coriolis: gtscript.Field[(dtype, (dim,))], 
    
    g: float,
    radius: float,
    determ: float,
    bd_det_x: float,
    bd_det_y: float,

    # Boundary
    phi_bd_N: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_S: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_E: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_W: gtscript.Field[(dtype, (n_qp_1D, dim))],

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
    hv_w: gtscript.Field[(dtype, (n_qp_1D,))]
):
    with computation(PARALLEL), interval(...):
        # Flux Stencil
        h_qp = phi @ h
        hu_qp = phi @ hu
        hv_qp = phi @ hv

        fh_x = hu_qp
        fh_y = hv_qp

        fhu_x = hu_qp * hu_qp / h_qp + h_qp * h_qp * g / 2
        fhu_y = hu_qp * hv_qp / h_qp

        fhv_x = hu_qp * hv_qp / h_qp
        fhv_y = hv_qp * hv_qp / h_qp + h_qp * h_qp * g / 2

        rhs_h = (phi_grad_x.T @ (fh_x * w) / bd_det_x + phi_grad_y.T @ (fh_y * w * cos_fact) / bd_det_y) * determ
        rhs_hu = (phi_grad_x.T @ (fhu_x * w) / bd_det_x + phi_grad_y.T @ (fhu_y * w * cos_fact) / bd_det_y) * determ
        rhs_hv = (phi_grad_x.T @ (fhv_x * w) / bd_det_x + phi_grad_y.T @ (fhv_y * w * cos_fact) / bd_det_y) * determ

        # Source Stencil
        rhs_hv -= (phi.T @ (0.5 * g * sin_fact * h_qp *h_qp * w)) * determ
        # Coriolis
        rhs_hu += (phi.T @ (coriolis * cos_fact * hv_qp * w)) * radius * determ
        rhs_hv -= (phi.T @ (coriolis * cos_fact * hu_qp * w)) * radius * determ

        # Boundary
        h_n = phi_bd_N @ h
        h_s = phi_bd_S @ h
        h_e = phi_bd_E @ h
        h_w = phi_bd_W @ h

        hu_n = phi_bd_N @ hu
        hu_s = phi_bd_S @ hu
        hu_e = phi_bd_E @ hu
        hu_w = phi_bd_W @ hu

        hv_n = phi_bd_N @ hv
        hv_s = phi_bd_S @ hv
        hv_e = phi_bd_E @ hv
        hv_w = phi_bd_W @ hv


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

    cos_fact: gtscript.Field[(dtype, (dim,))], 
    
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

        fhu_x = hu_qp * hu_qp / h_qp + h_qp * h_qp * g / 2
        fhu_y = hu_qp * hv_qp / h_qp

        fhv_x = hu_qp * hv_qp / h_qp
        fhv_y = hv_qp * hv_qp / h_qp + h_qp * h_qp * g / 2

        rhs_h = (phi_grad_x.T @ (fh_x * w) / bd_det_x + phi_grad_y.T @ (fh_y * w * cos_fact) / bd_det_y) * determ
        rhs_hu = (phi_grad_x.T @ (fhu_x * w) / bd_det_x + phi_grad_y.T @ (fhu_y * w * cos_fact) / bd_det_y) * determ
        rhs_hv = (phi_grad_x.T @ (fhv_x * w) / bd_det_x + phi_grad_y.T @ (fhv_y * w * cos_fact) / bd_det_y) * determ

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
def fused_num_flux(
    # Num Flux
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

    flux_n_h: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_s_h: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_e_h: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_w_h: gtscript.Field[(dtype, (n_qp_1D,))],

    flux_n_hu: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_s_hu: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_e_hu: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_w_hu: gtscript.Field[(dtype, (n_qp_1D,))],

    flux_n_hv: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_s_hv: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_e_hv: gtscript.Field[(dtype, (n_qp_1D,))],
    flux_w_hv: gtscript.Field[(dtype, (n_qp_1D,))],

    cos_n: gtscript.Field[(dtype, (n_qp_1D,))],
    cos_s: gtscript.Field[(dtype, (n_qp_1D,))],
    alpha: float,

    # Integrate Num Flux
    phi_bd_N: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_S: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_E: gtscript.Field[(dtype, (n_qp_1D, dim))],
    phi_bd_W: gtscript.Field[(dtype, (n_qp_1D, dim))],

    rhs_h: gtscript.Field[(dtype, (dim,))],
    rhs_hu: gtscript.Field[(dtype, (dim,))],
    rhs_hv: gtscript.Field[(dtype, (dim,))],
    tmp: gtscript.Field[(dtype, (dim,))],

    w: gtscript.Field[(dtype, (n_qp_1D,))],

    bd_det_x: float,
    bd_det_y: float,

    # Inv Mass
    inv_mass: gtscript.Field[(dtype, (dim, dim))],
    radius: float
):
    with computation(PARALLEL), interval(...):
        # --- Num Flux ---
        flux_n_h = cos_n * (0.5 * (f_n_h + f_s_h[0,+1,0]) - (h_s[0,+1,0] - h_n) * 0.5 * alpha)
        flux_s_h = cos_s * (-0.5 * (f_s_h + f_n_h[0,-1,0]) - (h_n[0,-1,0] - h_s) * 0.5 * alpha)
        flux_e_h = 0.5 * (f_e_h + f_w_h[+1,0,0]) - (h_w[+1,0,0] - h_e) * 0.5 * alpha
        flux_w_h = -0.5 * (f_w_h + f_e_h[-1,0,0]) -  (h_e[-1,0,0] - h_w) * 0.5 * alpha

        flux_n_hu = cos_n * (0.5 * (f_n_hu + f_s_hu[0,+1,0]) - (hu_s[0,+1,0] - hu_n) * 0.5 * alpha)
        flux_s_hu = cos_s * (-0.5 * (f_s_hu + f_n_hu[0,-1,0]) - (hu_n[0,-1,0] - hu_s) * 0.5 * alpha)
        flux_e_hu = 0.5 * (f_e_hu + f_w_hu[+1,0,0]) - (hu_w[+1,0,0] - hu_e) * 0.5 * alpha
        flux_w_hu = -0.5 * (f_w_hu + f_e_hu[-1,0,0]) -  (hu_e[-1,0,0] - hu_w) * 0.5 * alpha

        flux_n_hv = cos_n * (0.5 * (f_n_hv + f_s_hv[0,+1,0]) - (hv_s[0,+1,0] - hv_n) * 0.5 * alpha)
        flux_s_hv = cos_s * (-0.5 * (f_s_hv + f_n_hv[0,-1,0]) - (hv_n[0,-1,0] - hv_s) * 0.5 * alpha)
        flux_e_hv = 0.5 * (f_e_hv + f_w_hv[+1,0,0]) - (hv_w[+1,0,0] - hv_e) * 0.5 * alpha
        flux_w_hv = -0.5 * (f_w_hv + f_e_hv[-1,0,0]) -  (hv_e[-1,0,0] - hv_w) * 0.5 * alpha

        # --- Integrate Num Flux ---
        rhs_h -= (phi_bd_N.T @ (flux_n_h * w)) * bd_det_x
        rhs_h -= (phi_bd_S.T @ (flux_s_h * w)) * bd_det_x
        rhs_h -= (phi_bd_E.T @ (flux_e_h * w)) * bd_det_y
        rhs_h -= (phi_bd_W.T @ (flux_w_h * w)) * bd_det_y

        rhs_hu -= (phi_bd_N.T @ (flux_n_hu * w)) * bd_det_x
        rhs_hu -= (phi_bd_S.T @ (flux_s_hu * w)) * bd_det_x
        rhs_hu -= (phi_bd_E.T @ (flux_e_hu * w)) * bd_det_y
        rhs_hu -= (phi_bd_W.T @ (flux_w_hu * w)) * bd_det_y

        rhs_hv -= (phi_bd_N.T @ (flux_n_hv * w)) * bd_det_x
        rhs_hv -= (phi_bd_S.T @ (flux_s_hv * w)) * bd_det_x
        rhs_hv -= (phi_bd_E.T @ (flux_e_hv * w)) * bd_det_y
        rhs_hv -= (phi_bd_W.T @ (flux_w_hv * w)) * bd_det_y

        # --- Inv Mass ---
        # f_n_h(u/v) used as tmp
        tmp = (inv_mass @ rhs_h) / radius
        rhs_h = tmp
        tmp = (inv_mass @ rhs_hu) / radius
        rhs_hu = tmp
        tmp = (inv_mass @ rhs_hv) / radius
        rhs_hv = tmp

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

    cos_n: gtscript.Field[(dtype, (n_qp_1D,))],
    cos_s: gtscript.Field[(dtype, (n_qp_1D,))],

    alpha: float
):
    with computation(PARALLEL), interval(...):
        # BUG: alpha needs to be at end
        flux_n = cos_n * (0.5 * (f_n + f_s[0,+1,0]) - (u_s[0,+1,0] - u_n) * 0.5 * alpha)
        flux_s = cos_s * (-0.5 * (f_s + f_n[0,-1,0]) - (u_n[0,-1,0] - u_s) * 0.5 * alpha)
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

@gtscript.stencil(backend=backend, **backend_opts)
def source_stencil(
    phi: gtscript.Field[(dtype, (n_qp, dim))],

    h_qp: gtscript.Field[(dtype, (n_qp,))],
    sin_fact: gtscript.Field[(dtype, (dim,))],

    rhs_hv: gtscript.Field[(dtype, (dim,))],

    w: gtscript.Field[(dtype, (n_qp,))],
    g: float,
    determ: float
):
    with computation(PARALLEL), interval(...):
        rhs_hv -= (phi.T @ (0.5 * g * sin_fact * h_qp *h_qp * w)) * determ

@gtscript.stencil(backend=backend, **backend_opts)
def coriolis_stencil(
    phi: gtscript.Field[(dtype, (n_qp, dim))],

    coriolis: gtscript.Field[(dtype, (n_qp,))],

    hu_qp: gtscript.Field[(dtype, (n_qp,))],
    hv_qp: gtscript.Field[(dtype, (n_qp,))],
    cos_fact: gtscript.Field[(dtype, (dim,))],

    rhs_hu: gtscript.Field[(dtype, (dim,))],
    rhs_hv: gtscript.Field[(dtype, (dim,))],

    w: gtscript.Field[(dtype, (n_qp,))],
    radius: float,
    determ: float
):
    with computation(PARALLEL), interval(...):
        rhs_hu += (phi.T @ (coriolis * cos_fact * hv_qp * w)) * radius * determ
        rhs_hv -= (phi.T @ (coriolis * cos_fact * hu_qp * w)) * radius * determ


@gtscript.stencil(backend=backend, **backend_opts)
def inv_mass_stencil(
    rhs_h: gtscript.Field[(dtype, (dim,))],
    rhs_hu: gtscript.Field[(dtype, (dim,))],
    rhs_hv: gtscript.Field[(dtype, (dim,))],

    tmp: gtscript.Field[(dtype, (dim,))],

    inv_mass: gtscript.Field[(dtype, (dim, dim))],

    radius: float
):
    with computation(PARALLEL), interval(...):
        tmp = (inv_mass @ rhs_h) / radius
        rhs_h = tmp
        tmp = (inv_mass @ rhs_hu) / radius
        rhs_hu = tmp
        tmp = (inv_mass @ rhs_hv) / radius
        rhs_hv = tmp
        
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

@gtscript.stencil(backend=backend, **backend_opts)
def apply_pbc_east(
    h_n_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_n_right: gtscript.Field[(dtype, (n_qp_1D,))],
    h_s_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_s_right: gtscript.Field[(dtype, (n_qp_1D,))],
    h_e_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_e_right: gtscript.Field[(dtype, (n_qp_1D,))],
    h_w_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_w_right: gtscript.Field[(dtype, (n_qp_1D,))],

    hu_n_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_n_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_s_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_s_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_e_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_e_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_w_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_w_right: gtscript.Field[(dtype, (n_qp_1D,))],

    hv_n_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_n_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_s_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_s_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_e_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_e_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_w_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_w_right: gtscript.Field[(dtype, (n_qp_1D,))],
):
    with computation(PARALLEL), interval(...):
        h_n_left = h_n_right[-1,0,0]
        h_s_left = h_s_right[-1,0,0]
        h_e_left = h_e_right[-1,0,0]
        h_w_left = h_w_right[-1,0,0]

        hu_n_left = hu_n_right[-1,0,0]
        hu_s_left = hu_s_right[-1,0,0]
        hu_e_left = hu_e_right[-1,0,0]
        hu_w_left = hu_w_right[-1,0,0]

        hv_n_left = hv_n_right[-1,0,0]
        hv_s_left = hv_s_right[-1,0,0]
        hv_e_left = hv_e_right[-1,0,0]
        hv_w_left = hv_w_right[-1,0,0]

@gtscript.stencil(backend=backend, **backend_opts)
def apply_pbc_west(
    h_n_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_n_right: gtscript.Field[(dtype, (n_qp_1D,))],
    h_s_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_s_right: gtscript.Field[(dtype, (n_qp_1D,))],
    h_e_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_e_right: gtscript.Field[(dtype, (n_qp_1D,))],
    h_w_left: gtscript.Field[(dtype, (n_qp_1D,))],
    h_w_right: gtscript.Field[(dtype, (n_qp_1D,))],

    hu_n_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_n_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_s_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_s_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_e_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_e_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_w_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hu_w_right: gtscript.Field[(dtype, (n_qp_1D,))],

    hv_n_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_n_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_s_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_s_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_e_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_e_right: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_w_left: gtscript.Field[(dtype, (n_qp_1D,))],
    hv_w_right: gtscript.Field[(dtype, (n_qp_1D,))],
):
    with computation(PARALLEL), interval(...):
        h_n_right = h_n_left[+1,0,0]
        h_s_right = h_s_left[+1,0,0]
        h_e_right = h_e_left[+1,0,0]
        h_w_right = h_w_left[+1,0,0]

        hu_n_right = hu_n_left[+1,0,0]
        hu_s_right = hu_s_left[+1,0,0]
        hu_e_right = hu_e_left[+1,0,0]
        hu_w_right = hu_w_left[+1,0,0]

        hv_n_right = hv_n_left[+1,0,0]
        hv_s_right = hv_s_left[+1,0,0]
        hv_e_right = hv_e_left[+1,0,0]
        hv_w_right = hv_w_left[+1,0,0]