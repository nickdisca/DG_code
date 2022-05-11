import numpy as np
import time
import gt4py.gtscript as gtscript
import gt4py as gt

from gt4py_config import dtype, backend, backend_opts, runge_kutta

import stencils
import boundary_conditions

def run(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp1d, n_qp2d, hx, hy, nx, ny, dx, dt, niter, plotter):
    alpha = dx / dt
    determ = hx * hy / 4
    bd_det_x = hx / 2
    bd_det_y = hy / 2
    radius=6.37122e6
    nz = 1
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

    u_nodal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,))) # for plotting

    # --- runge kutta --- 
    k1 = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k2 = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k3 = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (dim,)))
    k4 = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
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
    ## NOTE Default origin is NOT (0,0,0)
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

    tmp = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
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
                wts2d, wts1d, nx, ny, alpha, radius
            )
            # --- Timestepping ---
            stencils.rk_step1(rhs_h, h, dt, h)
            stencils.rk_step1(rhs_hv, hu, dt, hu)
            stencils.rk_step1(rhs_hv, hv, dt, hv)
        # elif runge_kutta == 2:
        #     compute_rhs(
        #         uM_gt, rhs, u_qp, fx, fy, u_n, u_s, u_e, u_w,
        #         f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
        #         determ, bd_det_x, bd_det_y, vander, inv_mass,
        #         wts2d, wts1d, nx, ny, dt
        #     )
        #     stencils.rk_step1(rhs, uM_gt, dt, k1)
        #     compute_rhs(
        #         k1, k2, u_qp, fx, fy, u_n, u_s, u_e, u_w,
        #         f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
        #         determ, bd_det_x, bd_det_y, vander, inv_mass,
        #         wts2d, wts1d, nx, ny, dt
        #     )
        #     stencils.rk_step2(k1, k2, uM_gt, dt, uM_gt)
        # elif runge_kutta == 3:
        #     compute_rhs(
        #         uM_gt, rhs, u_qp, fx, fy, u_n, u_s, u_e, u_w,
        #         f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
        #         determ, bd_det_x, bd_det_y, vander, inv_mass,
        #         wts2d, wts1d, nx, ny, dt
        #     )
        #     stencils.rk_step1(rhs, uM_gt, dt, k1)
        #     compute_rhs(
        #         k1, k2, u_qp, fx, fy, u_n, u_s, u_e, u_w,
        #         f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
        #         determ, bd_det_x, bd_det_y, vander, inv_mass,
        #         wts2d, wts1d, nx, ny, dt
        #     )
        #     stencils.rk_step2_3(k1, k2, uM_gt, dt, uM_gt)
        #     compute_rhs(
        #         k2, k3, u_qp, fx, fy, u_n, u_s, u_e, u_w,
        #         f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
        #         determ, bd_det_x, bd_det_y, vander, inv_mass,
        #         wts2d, wts1d, nx, ny, dt
        #     )
        #     stencils.rk_step3_3(k2, k3, uM_gt, dt, uM_gt)


        # --- Output --- 
        print(f'Iteration {i} done')
        if i % plot_freq == 0:
            stencils.modal2nodal(vander.vander_gt, h, u_nodal)
            plotter.plot_solution(u_nodal, init=False, plot_type=plot_type)

    loop_end = time.perf_counter()

    print('--- Timings ---')
    print(f'Loop: {loop_end - loop_start}s')
    print(f'Allocation: {alloc_end - alloc_start}s')



def compute_rhs(
    cons_var, rhs, cons_qp, fx, fy, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
    flux_n, flux_s, flux_e, flux_w, 
    determ, bd_det_x, bd_det_y,
    vander, inv_mass, wts2d, wts1d, nx, ny, alpha, radius
):
        g = 9.81
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

        # --- Flux Integral ---
        stencils.flux_stencil_swe(
            vander.phi_gt, h, hu, hv, h_qp, hu_qp, hv_qp, 
            fh_x, fh_y, fhu_x, fhu_y, fhv_x, fhv_y,
            vander.grad_phi_x_gt, vander.grad_phi_y_gt, wts2d,
            rhs_h, rhs_hu, rhs_hv, g, determ, bd_det_x, bd_det_y
        )

        # --- Boundary Integral ---
        origins = {
            "_all_": (0,0,0),'u_n': (1,1,0), 'u_s': (1,1,0), 'u_e': (1,1,0), 'u_w': (1,1,0)
        }
        stencils.modal2bd(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, h_n, h_s, h_e, h_w, h,
            origin=origins, domain=(nx,ny,1)
        )
        boundary_conditions.apply_pbc(h_n)
        boundary_conditions.apply_pbc(h_s)
        boundary_conditions.apply_pbc(h_e)
        boundary_conditions.apply_pbc(h_w)

        stencils.modal2bd(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, hu_n, hu_s, hu_e, hu_w, hu,
            origin=origins, domain=(nx,ny,1)
        )
        boundary_conditions.apply_pbc(hu_n)
        boundary_conditions.apply_pbc(hu_s)
        boundary_conditions.apply_pbc(hu_e)
        boundary_conditions.apply_pbc(hu_w)

        stencils.modal2bd(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, hv_n, hv_s, hv_e, hv_w, hv,
            origin=origins, domain=(nx,ny,1)
        )
        boundary_conditions.apply_pbc(hv_n)
        boundary_conditions.apply_pbc(hv_s)
        boundary_conditions.apply_pbc(hv_e)
        boundary_conditions.apply_pbc(hv_w)

        stencils.flux_bd_stencil_swe(
            h_n, h_s, h_e, h_w, hu_n, hu_s, hu_e, hu_w,
            hv_n, hv_s, hv_e, hv_w, f_n_h, f_s_h, f_e_h, f_w_h,
            f_n_hu, f_s_hu, f_e_hu, f_w_hu, f_n_hv, f_s_hv, f_e_hv, f_w_hv,
            origin=(0,0,0), domain=(nx+2, ny+2, 1)
        )

        origins = {
            "_all_": (1,1,0),'flux_n': (0,0,0), 'flux_s': (0,0,0), 'flux_e': (0,0,0), 'flux_w': (0,0,0)
        }
        stencils.compute_num_flux(
            h_n, h_s, h_e, h_w, f_n_h, f_s_h, f_e_h, f_w_h,
            flux_n_h, flux_s_h, flux_e_h, flux_w_h, alpha,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.compute_num_flux(
            hu_n, hu_s, hu_e, hu_w, f_n_hu, f_s_hu, f_e_hu, f_w_hu,
            flux_n_hu, flux_s_hu, flux_e_hu, flux_w_hu, alpha,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.compute_num_flux(
            hv_n, hv_s, hv_e, hv_w, f_n_hv, f_s_hv, f_e_hv, f_w_hv,
            flux_n_hv, flux_s_hv, flux_e_hv, flux_w_hv, alpha,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.integrate_num_flux(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, flux_n_h, flux_s_h, flux_e_h, flux_w_h, wts1d, rhs_h,
            inv_mass, radius, bd_det_x, bd_det_y
        )

        stencils.integrate_num_flux(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, flux_n_hu, flux_s_hu, flux_e_hu, flux_w_hu, wts1d, rhs_hu,
            inv_mass, radius, bd_det_x, bd_det_y
        )

        stencils.integrate_num_flux(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, flux_n_hv, flux_s_hv, flux_e_hv, flux_w_hv, wts1d, rhs_hv,
            inv_mass, radius, bd_det_x, bd_det_y
        )