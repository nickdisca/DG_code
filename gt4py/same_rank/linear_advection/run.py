import numpy as np
import time
import gt4py.gtscript as gtscript
import gt4py as gt

from gt4py_config import dtype, backend, backend_opts, runge_kutta

import stencils
import boundary_conditions

def run(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp1d, n_qp2d, hx, hy, nx, ny, alpha, dt, niter, plotter):
    determ = hx * hy / 4
    bd_det_x = hx / 2
    bd_det_y = hy / 2
    radius = 1
    nz = 1
    plot_freq = plotter.plot_freq
    plot_type = plotter.plot_type

    # === Memory allocation ===
    alloc_start = time.perf_counter()

    rhs = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
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
    u_qp = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fx = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    fy = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))

    # --- boundary integrals ---
    ## NOTE Default origin is NOT (0,0,0)
    u_n = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    u_s = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    u_e = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    u_w = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_n = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_s = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_e = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    f_w = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))
    tmp = gt.storage.zeros(backend=backend, default_origin=(1,1,0),
        shape=(nx+2, ny+2, nz), dtype=(dtype, (n_qp1d,)))

    flux_n = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_s = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_e = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    flux_w = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp1d,)))
    alloc_end = time.perf_counter()
    # === End ===

    loop_start = time.perf_counter()
    for i in range(niter):
        if runge_kutta == 1:
            compute_rhs(
                uM_gt, rhs, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            # --- Timestepping ---
            stencils.rk_step1(rhs, uM_gt, dt, uM_gt)
        elif runge_kutta == 2:
            compute_rhs(
                uM_gt, rhs, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step1(rhs, uM_gt, dt, k1) # computes k1 = u_bar
            compute_rhs(
                k1, k2, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            # stencils.rk_step2(rhs, k2, uM_gt, dt, uM_gt)
            stencils.rk_step2_paper(k1, k2, uM_gt, dt, uM_gt)
        elif runge_kutta == 3:
            compute_rhs(
                uM_gt, k1, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step1(k1, uM_gt, dt, rhs)
            compute_rhs(
                rhs, k2, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step2_3(k1, k2, uM_gt, dt, rhs)
            compute_rhs(
                rhs, k3, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step3_3(k1, k2, k3, uM_gt, dt, uM_gt)
        elif runge_kutta == 4:
            compute_rhs(
                uM_gt, k1, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step1_4(k1, uM_gt, dt, rhs)
            compute_rhs(
                rhs, k2, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step1_4(k2, uM_gt, dt, rhs)
            compute_rhs(
                rhs, k3, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step1(k3, uM_gt, dt, rhs)
            compute_rhs(
                rhs, k4, u_qp, fx, fy, u_n, u_s, u_e, u_w,
                f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w,
                determ, bd_det_x, bd_det_y, vander, inv_mass,
                wts2d, wts1d, nx, ny, dt, alpha
            )
            stencils.rk_step2_4(k1, k2, k3, k4, uM_gt, dt, uM_gt)


        # === OUTPUT DONE === 
        # print(f'Iteration {i} done')
        if i % plot_freq == 0:
            stencils.modal2nodal(vander.vander_gt, uM_gt, u_nodal)
            plotter.plot_solution(u_nodal, init=False, plot_type=plot_type)
        # === OUTPUT DONE ===

    loop_end = time.perf_counter()

    print('--- Timings ---')
    print(f'Loop: {loop_end - loop_start}s')
    print(f'Allocation: {alloc_end - alloc_start}s')



def compute_rhs(
    uM_gt, rhs, u_qp, fx, fy, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
    flux_n, flux_s, flux_e, flux_w, 
    determ, bd_det_x, bd_det_y,
    vander, inv_mass, wts2d, wts1d, nx, ny, dt, alpha
):
        # --- Flux Integral ---
        stencils.flux_stencil(
            vander.phi_gt, uM_gt, u_qp, fx, fy, vander.grad_phi_x_gt,
            vander.grad_phi_y_gt, wts2d, rhs, determ, bd_det_x, bd_det_y
        )

        # --- Boundary Integral ---
        origins = {
            "_all_": (0,0,0),'u_n': (1,1,0), 'u_s': (1,1,0), 'u_e': (1,1,0), 'u_w': (1,1,0)
        }
        stencils.modal2bd(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, u_n, u_s, u_e, u_w, uM_gt,
            origin=origins, domain=(nx,ny,1)
        )
        boundary_conditions.apply_pbc(u_n)
        boundary_conditions.apply_pbc(u_s)
        boundary_conditions.apply_pbc(u_e)
        boundary_conditions.apply_pbc(u_w)

        stencils.flux_bd_stencil(
            u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
            origin=(0,0,0), domain=(nx+2, ny+2, 1)
        )

        origins = {
            "_all_": (1,1,0),'flux_n': (0,0,0), 'flux_s': (0,0,0), 'flux_e': (0,0,0), 'flux_w': (0,0,0)
        }
        stencils.compute_num_flux(
            u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
            flux_n, flux_s, flux_e, flux_w, alpha,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.integrate_num_flux(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, flux_n, flux_s, flux_e, flux_w, wts1d, rhs,
            inv_mass, bd_det_x, bd_det_y
        )
