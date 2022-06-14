import numpy as np
import time
import gt4py as gt

from gt4py_config import dtype, backend, runge_kutta

import stencils
import boundary_conditions

def run(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp1d, n_qp2d, hx, hy, nx, ny, nz, cos_fact, sin_fact, cos_bd, coriolis, radius, alpha, dt, niter, plotter):
    determ = hx * hy / 4
    bd_det_x = hx / 2
    bd_det_y = hy / 2
    plot_freq = plotter.plot_freq
    plot_type = plotter.plot_type

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
            # # k1_* serve as temps
            # stencils.modal2nodal(vander.vander_gt, h, k1_h)
            # if np.max(np.abs(k1_h)) > 1e8:
            #     raise Exception('Solution diverging')
            # stencils.modal2nodal(vander.vander_gt, hu, k1_hu)
            # stencils.modal2nodal(vander.vander_gt, hv, k1_hv)
            # print('plotting')
            # plotter.plot_solution((k1_h, k1_hu, k1_hv), title=f'{i * dt:.1f}s ({i * dt / 3600:.1f} h {i * dt / 86400:.1f} days)', init=False, fname=f'{i}.png',  plot_type=plot_type)

    loop_end = time.perf_counter()

    print('--- Timings ---')
    print(f'Loop: {loop_end - loop_start}s')
    print(f'Allocation: {alloc_end - alloc_start}s')



def compute_rhs(
    cons_var, rhs, cons_qp, fx, fy, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
    flux_n, flux_s, flux_e, flux_w, 
    determ, bd_det_x, bd_det_y,
    vander, inv_mass, cos_fact, sin_fact, cos_bd, coriolis, tmp,
    wts2d, wts1d, nx, ny, nz, alpha, radius
):
        g = 9.80616
        h, hu, hv = cons_var
        rhs_h, rhs_hu, rhs_hv = rhs
        h_qp, hu_qp, hv_qp = cons_qp
        fh_x, fhu_x, fhv_x = fx
        fh_y, fhu_y, fhv_y = fy

        h_n, hu_n, hv_n = u_n
        h_s, hu_s, hv_s = u_s
        h_e, hu_e, hv_e = u_e
        h_w, hu_w, hv_w = u_w

        f_n_h, f_n_hu, f_n_hv = f_n
        f_s_h, f_s_hu, f_s_hv = f_s
        f_e_h, f_e_hu, f_e_hv = f_e
        f_w_h, f_w_hu, f_w_hv = f_w

        flux_n_h, flux_n_hu, flux_n_hv = flux_n
        flux_s_h, flux_s_hu, flux_s_hv = flux_s
        flux_e_h, flux_e_hu, flux_e_hv = flux_e
        flux_w_h, flux_w_hu, flux_w_hv = flux_w

        cos_n, cos_s = cos_bd

        origins = {
            '_all_': (0,0,0),'h_n': (1,1,0), 'h_s': (1,1,0), 'h_e': (1,1,0), 'h_w': (1,1,0),
            'hu_n': (1,1,0), 'hu_s': (1,1,0), 'hu_e': (1,1,0), 'hu_w': (1,1,0),
            'hv_n': (1,1,0), 'hv_s': (1,1,0), 'hv_e': (1,1,0), 'hv_w': (1,1,0)
        }
        # --- Internal Fused---
        stencils.fused_internal_stencils(
            vander.phi_gt, h, hu, hv, h_qp, hu_qp, hv_qp, 
            fh_x, fh_y, fhu_x, fhu_y, fhv_x, fhv_y,
            vander.grad_phi_x_gt, vander.grad_phi_y_gt, wts2d,
            rhs_h, rhs_hu, rhs_hv, cos_fact, sin_fact, coriolis,
            g, radius, determ, bd_det_x, bd_det_y,

            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt, vander.phi_bd_W_gt,
            h_n, h_s, h_e, h_w,
            hu_n, hu_s, hu_e, hu_w,
            hv_n, hv_s, hv_e, hv_w,
        )
        h_qp.device_to_host()
        hu_qp.device_to_host()
        hv_qp.device_to_host()

        h_qp_tmp = np.asarray(h_qp)
        hu_qp_tmp = np.asarray(hu_qp)
        hv_qp_tmp = np.asarray(hv_qp)
        print(f'{h_qp_tmp = }\n{hu_qp_tmp = }\n{hv_qp_tmp = }')
        quit()


        # # --- Internal NOT Fused--- 
        # stencils.flux_stencil_swe(
        #     vander.phi_gt, h, hu, hv, h_qp, hu_qp, hv_qp, 
        #     fh_x, fh_y, fhu_x, fhu_y, fhv_x, fhv_y,
        #     vander.grad_phi_x_gt, vander.grad_phi_y_gt, wts2d,
        #     rhs_h, rhs_hu, rhs_hv, cos_fact, g, determ, bd_det_x, bd_det_y
        # )

        # stencils.source_stencil(
        #     vander.phi_gt, h_qp, sin_fact, rhs_hv, wts2d, g, determ
        # )

        # stencils.coriolis_stencil(
        #     vander.phi_gt, coriolis, hu_qp, hv_qp, cos_fact, rhs_hu, rhs_hv, wts2d, radius, determ
        # )

        # # --- Boundary NOT Fused ---
        # origins = {
        #     "_all_": (0,0,0),'u_n': (1,1,0), 'u_s': (1,1,0), 'u_e': (1,1,0), 'u_w': (1,1,0)
        # }
        # stencils.modal2bd(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, h_n, h_s, h_e, h_w, h,
        #     origin=origins, domain=(nx,ny,nz)
        # )
        # stencils.modal2bd(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, hu_n, hu_s, hu_e, hu_w, hu,
        #     origin=origins, domain=(nx,ny,nz)
        # )
        # stencils.modal2bd(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, hv_n, hv_s, hv_e, hv_w, hv,
        #     origin=origins, domain=(nx,ny,nz)
        # )
        # --- PBC Stencil ---
        origins_pbc = {
            "h_e_left": (0,1,0), "h_e_right": (nx+1, 1, 0),
            "h_w_left": (0,1,0), "h_w_right": (nx+1, 1, 0),

            "hu_e_left": (0,1,0), "hu_e_right": (nx+1, 1, 0),
            "hu_w_left": (0,1,0), "hu_w_right": (nx+1, 1, 0),

            "hv_e_left": (0,1,0), "hv_e_right": (nx+1, 1, 0),
            "hv_w_left": (0,1,0), "hv_w_right": (nx+1, 1, 0),

            "h_n_bot": (1,0,0), "h_n_top": (1, ny+1, 0),
            "h_s_bot": (1,0,0), "h_s_top": (1, ny+1, 0),
            "hu_n_bot": (1,0,0), "hu_n_top": (1, ny+1, 0),
            "hu_s_bot": (1,0,0), "hu_s_top": (1, ny+1, 0),
            "hv_n_bot": (1,0,0), "hv_n_top": (1, ny+1, 0),
            "hv_s_bot": (1,0,0), "hv_s_top": (1, ny+1, 0),

        }
        stencils.apply_pbc_east(
            h_e, h_e, h_w, h_w,
            hu_e, hu_e, hu_w, hu_w,
            hv_e, hv_e, hv_w, hv_w,
            origin=origins_pbc, domain=(1, ny, nz)
        )
        stencils.apply_pbc_west(
            h_e, h_e, h_w, h_w,
            hu_e, hu_e, hu_w, hu_w,
            hv_e, hv_e, hv_w, hv_w,
            origin=origins_pbc, domain=(1, ny, nz)
        )
        # NOTE: north/south pbc not stricly necessary
        stencils.apply_pbc_south(
            h_n, h_n, h_s, h_s,
            hu_n, hu_n, hu_s, hu_s,
            hv_n, hv_n, hv_s, hv_s,
            origin=origins_pbc, domain=(nx, 1, nz)
        )
        stencils.apply_pbc_north(
            h_n, h_n, h_s, h_s,
            hu_n, hu_n, hu_s, hu_s,
            hv_n, hv_n, hv_s, hv_s,
            origin=origins_pbc, domain=(nx, 1, nz)
        )
        # --- PBC function ---

        # boundary_conditions.apply_pbc(h_n)
        # boundary_conditions.apply_pbc(h_s)
        # boundary_conditions.apply_pbc(h_e)
        # boundary_conditions.apply_pbc(h_w)

        # boundary_conditions.apply_pbc(hv_n)
        # boundary_conditions.apply_pbc(hv_s)
        # boundary_conditions.apply_pbc(hv_e)
        # boundary_conditions.apply_pbc(hv_w)

        # boundary_conditions.apply_pbc(hu_n)
        # boundary_conditions.apply_pbc(hu_s)
        # boundary_conditions.apply_pbc(hu_e)
        # boundary_conditions.apply_pbc(hu_w)
        
        # --- 

        stencils.flux_bd_stencil_swe(
            h_n, h_s, h_e, h_w, hu_n, hu_s, hu_e, hu_w,
            hv_n, hv_s, hv_e, hv_w, f_n_h, f_s_h, f_e_h, f_w_h,
            f_n_hu, f_s_hu, f_e_hu, f_w_hu, f_n_hv, f_s_hv, f_e_hv, f_w_hv, g,
            origin=(0,0,0), domain=(nx+2, ny+2, nz)
        )
        # --- Num Flux Fused ---
        origins = {
            "_all_": (1,1,0),
            "flux_n_h": (0,0,0), "flux_s_h": (0,0,0), "flux_e_h": (0,0,0), "flux_w_h": (0,0,0),
            "flux_n_hu": (0,0,0), "flux_s_hu": (0,0,0), "flux_e_hu": (0,0,0), "flux_w_hu": (0,0,0),
            "flux_n_hv": (0,0,0), "flux_s_hv": (0,0,0), "flux_e_hv": (0,0,0), "flux_w_hv": (0,0,0),
            "phi_bd_N": (0,0,0), "phi_bd_S": (0,0,0), "phi_bd_E": (0,0,0), "phi_bd_W": (0,0,0),
            "rhs_h": (0,0,0), "rhs_hu": (0,0,0), "rhs_hv": (0,0,0), "tmp": (0,0,0), "w": (0,0,0),
            "inv_mass": (0,0,0), "cos_n": (0,0,0), "cos_s": (0,0,0)
        }
        stencils.fused_num_flux(
            h_n, h_s, h_e, h_w,
            hu_n, hu_s, hu_e, hu_w,
            hv_n, hv_s, hv_e, hv_w,

            f_n_h, f_s_h, f_e_h, f_w_h,
            f_n_hu, f_s_hu, f_e_hu, f_w_hu,
            f_n_hv, f_s_hv, f_e_hv, f_w_hv,

            flux_n_h, flux_s_h, flux_e_h, flux_w_h,
            flux_n_hu, flux_s_hu, flux_e_hu, flux_w_hu,
            flux_n_hv, flux_s_hv, flux_e_hv, flux_w_hv,
            cos_n, cos_s, alpha,

            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt, vander.phi_bd_W_gt,
            rhs_h, rhs_hu, rhs_hv, tmp, wts1d, bd_det_x, bd_det_y, 
            inv_mass, radius,
            origin=origins, domain=(nx,ny,nz)
        )

        # --- Num Flux NOT Fused ---
        # origins = {
        #     "_all_": (1,1,0),'flux_n': (0,0,0), 'flux_s': (0,0,0), 'flux_e': (0,0,0), 'flux_w': (0,0,0)
        # }

        # stencils.compute_num_flux(
        #     h_n, h_s, h_e, h_w, f_n_h, f_s_h, f_e_h, f_w_h,
        #     flux_n_h, flux_s_h, flux_e_h, flux_w_h, cos_n, cos_s, alpha,
        #     origin=origins, domain=(nx, ny, nz)
        # )

        # stencils.compute_num_flux(
        #     hu_n, hu_s, hu_e, hu_w, f_n_hu, f_s_hu, f_e_hu, f_w_hu,
        #     flux_n_hu, flux_s_hu, flux_e_hu, flux_w_hu, cos_n, cos_s, alpha,
        #     origin=origins, domain=(nx, ny, nz)
        # )

        # stencils.compute_num_flux(
        #     hv_n, hv_s, hv_e, hv_w, f_n_hv, f_s_hv, f_e_hv, f_w_hv,
        #     flux_n_hv, flux_s_hv, flux_e_hv, flux_w_hv, cos_n, cos_s, alpha,
        #     origin=origins, domain=(nx, ny, nz)
        # )

        # stencils.integrate_num_flux(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, flux_n_h, flux_s_h, flux_e_h, flux_w_h, wts1d, rhs_h,
        #     inv_mass, radius, bd_det_x, bd_det_y
        # )

        # stencils.integrate_num_flux(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, flux_n_hu, flux_s_hu, flux_e_hu, flux_w_hu, wts1d, rhs_hu,
        #     inv_mass, radius, bd_det_x, bd_det_y
        # )

        # stencils.integrate_num_flux(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, flux_n_hv, flux_s_hv, flux_e_hv, flux_w_hv, wts1d, rhs_hv,
        #     inv_mass, radius, bd_det_x, bd_det_y
        # )

        # stencils.inv_mass_stencil(rhs_h, rhs_hu, rhs_hv, tmp, inv_mass, radius)

