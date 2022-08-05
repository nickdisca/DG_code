import numpy as np
import time
import gt4py as gt

from gt4py_config import dtype, backend, runge_kutta

import stencils
from compute_rhs import compute_rhs

def run(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp1d, n_qp2d, hx, hy, nx, ny, nz, cos_fact, sin_fact, cos_bd, coriolis, radius, alpha, dt, niter, plotter):
    determ = hx * hy / 4
    bd_det_x = hx / 2
    bd_det_y = hy / 2
    plot_freq = plotter.plot_freq

    if type(uM_gt) == tuple:
        h, hu, hv = uM_gt

    # === Memory allocation ===
    alloc_start = time.perf_counter()

    rhs_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    rhs_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    rhs_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))

    tmp = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,))) # for plotting

    # --- runge kutta --- 
    k1_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k2_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k3_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k4_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))

    k1_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k2_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k3_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k4_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))

    k1_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k2_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k3_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k4_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
        
    # --- internal integrals ---
    h_qp = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fh_x = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fh_y = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))

    hu_qp = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fhu_x = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fhu_y = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))

    hv_qp = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fhv_x = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fhv_y = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))

    # --- boundary integrals ---
    ## NOTE: Padding -> Default origin is NOT (0,0,0)
    h_n = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    h_s = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    h_e = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    h_w = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))

    hu_n = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    hu_s = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    hu_e = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    hu_w = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))

    hv_n = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    hv_s = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    hv_e = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    hv_w = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
        
    f_n_h = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_s_h = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_e_h = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_w_h = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))

    f_n_hu = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_s_hu = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_e_hu = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_w_hu = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))

    f_n_hv = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_s_hv = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_e_hv = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_w_hv = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))

    flux_n_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_s_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_e_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_w_h = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))

    flux_n_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_s_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_e_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_w_hu = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))

    flux_n_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_s_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_e_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_w_hv = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    alloc_end = time.perf_counter()
    # === End ===

    loop_start = time.perf_counter()
    for i in range(niter):
        if runge_kutta == 1:
            compute_rhs(
                (h, hu, hv), (rhs_h, rhs_hu, rhs_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            # --- Timestepping ---
            stencils.rk_step1(rhs_h, h, dt, h)
            stencils.rk_step1(rhs_hu, hu, dt, hu)
            stencils.rk_step1(rhs_hv, hv, dt, hv)
        elif runge_kutta == 2:
            compute_rhs(
                (h, hu, hv), (k1_h, k1_hu, k1_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            # --- Timestepping ---
            stencils.rk_step1(k1_h, h, dt, rhs_h)
            stencils.rk_step1(k1_hu, hu, dt, rhs_hu)
            stencils.rk_step1(k1_hv, hv, dt, rhs_hv)
            compute_rhs(
                (rhs_h, rhs_hu, rhs_hv), (k2_h, k2_hu, k2_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            stencils.rk_step2(k1_h, k2_h, h, dt, h)
            stencils.rk_step2(k1_hu, k2_hu, hu, dt, hu)
            stencils.rk_step2(k1_hv, k2_hv, hv, dt, hv)
        elif runge_kutta == 3:
            compute_rhs(
                (h, hu, hv), (k1_h, k1_hu, k1_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            # --- Timestepping ---
            stencils.rk_step1(k1_h, h, dt, rhs_h)
            stencils.rk_step1(k1_hu, hu, dt, rhs_hu)
            stencils.rk_step1(k1_hv, hv, dt, rhs_hv)
            compute_rhs(
                (rhs_h, rhs_hu, rhs_hv), (k2_h, k2_hu, k2_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            stencils.rk_step2_3(k1_h, k2_h, h, dt, rhs_h)
            stencils.rk_step2_3(k1_hu, k2_hu, hu, dt, rhs_hu)
            stencils.rk_step2_3(k1_hv, k2_hv, hv, dt, rhs_hv)
            compute_rhs(
                (rhs_h, rhs_hu, rhs_hv), (k3_h, k3_hu, k3_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            stencils.rk_step3_3(k1_h, k2_h, k3_h, h, dt, h)
            stencils.rk_step3_3(k1_hu, k2_hu, k3_hu, hu, dt, hu)
            stencils.rk_step3_3(k1_hv, k2_hv, k3_hv, hv, dt, hv)
        elif runge_kutta == 4:
            compute_rhs(
                (h, hu, hv), (k1_h, k1_hu, k1_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            # --- Timestepping ---
            stencils.rk_step1_4(k1_h, h, dt, rhs_h)
            stencils.rk_step1_4(k1_hu, hu, dt, rhs_hu)
            stencils.rk_step1_4(k1_hv, hv, dt, rhs_hv)
            compute_rhs(
                (rhs_h, rhs_hu, rhs_hv), (k2_h, k2_hu, k2_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            stencils.rk_step1_4(k2_h, h, dt, rhs_h)
            stencils.rk_step1_4(k2_hu, hu, dt, rhs_hu)
            stencils.rk_step1_4(k2_hv, hv, dt, rhs_hv)
            compute_rhs(
                (rhs_h, rhs_hu, rhs_hv), (k3_h, k3_hu, k3_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            stencils.rk_step1(k3_h, h, dt, rhs_h)
            stencils.rk_step1(k3_hu, hu, dt, rhs_hu)
            stencils.rk_step1(k3_hv, hv, dt, rhs_hv)
            compute_rhs(
                (rhs_h, rhs_hu, rhs_hv), (k4_h, k4_hu, k4_hv), (h_qp, hu_qp, hv_qp), 
                (fh_x, fhu_x, fhv_x), (fh_y, fhu_y, fhv_y),
                (h_n, hu_n, hv_n), (h_s, hu_s, hv_s),
                (h_e, hu_e, hv_e), (h_w, hu_w, hv_w),
                (f_n_h, f_n_hu, f_n_hv), (f_s_h, f_s_hu, f_s_hv),
                (f_e_h, f_e_hu, f_e_hv), (f_w_h, f_w_hu, f_w_hv),
                (flux_n_h, flux_n_hv, flux_n_hu), (flux_s_h, flux_s_hu, flux_s_hv), (flux_e_h, flux_e_hv, flux_e_hu), (flux_w_h, flux_w_hv, flux_w_hu),
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                cos_fact, sin_fact, cos_bd, coriolis, tmp,
                wts2d, wts1d, nx, ny, nz, alpha, radius
            )
            stencils.rk_step2_4(k1_h, k2_h, k3_h, k4_h,  h, dt, h)
            stencils.rk_step2_4(k1_hu, k2_hu, k3_hu, k4_hu, hu, dt, hu)
            stencils.rk_step2_4(k1_hv, k2_hv, k3_hv, k4_hv, hv, dt, hv)

        # --- Output --- 
        if i % plot_freq == 0:
            print(f'Iteration {i}: time = {dt*i:.1f}s ({dt*i/3600:.1f} {dt*i/86400 :.1f} days)')
            # k1_* serves as temps
            stencils.modal2nodal(vander.vander_gt, h, k1_h)
            if np.max(np.abs(k1_h)) > 1e8:
                raise Exception('Solution diverging')

            stencils.modal2nodal(vander.vander_gt, hu, k1_hu)
            stencils.modal2nodal(vander.vander_gt, hv, k1_hv)
            print('plotting')
            plotter.plot_solution((k1_h, k1_hu, k1_hv), title=f'{i * dt:.1f}s ({i * dt / 3600:.1f} h {i * dt / 86400:.1f} days)', fname=f'simulation_{dt*i}')

    loop_end = time.perf_counter()

    print('--- Timings ---')
    print(f'Loop: {loop_end - loop_start}s')
    print(f'Allocation: {alloc_end - alloc_start}s')
