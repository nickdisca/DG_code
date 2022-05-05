import numpy as np
import time
import gt4py.gtscript as gtscript
import gt4py as gt

from gt4py_config import dtype, backend, backend_opts

import stencils
import boundary_conditions

def compute_rhs(uM_gt, vander, inv_mass, wts2d, wts1d, dim, n_qp1d, n_qp2d, hx, hy, nx, ny, dt, niter, plotter):
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
        # --- Flux Integrals ---

        # OG ---
        # modal2qp_stencil(vander.phi_gt, uM_gt, u_qp)
        # flux_function_stencil(u_qp, fx, fy)
        # integrate_flux(rhs, wts2d, fx, fy, vander, determ, bd_det_x, bd_det_y)

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
            flux_n, flux_s, flux_e, flux_w,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.integrate_num_flux(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, flux_n, flux_s, flux_e, flux_w, wts1d, rhs,
            bd_det_x, bd_det_y
        )

        # --- Timestepping ---
        stencils.rk_step(inv_mass, rhs, uM_gt, dt)


        # --- Output --- 
        print(f'Iteration {i} done')
        if i % plot_freq == 0:
            stencils.modal2nodal(vander.vander_gt, uM_gt, u_nodal)
            plotter.plot_solution(u_nodal, init=False, plot_type=plot_type)

    loop_end = time.perf_counter()

    print('--- Timings ---')
    print(f'Loop: {loop_end - loop_start}s')
    print(f'Allocation: {alloc_end - alloc_start}s')



