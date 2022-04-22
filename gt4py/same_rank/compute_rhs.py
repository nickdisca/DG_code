import numpy as np
import time
import gt4py.gtscript as gtscript
import gt4py as gt
from flux_function import flux_function_gt, integrate_flux_stencil, flux_function_stencil, complete_flux_stencil
from modal_conversion import modal2qp_gt, modal2bd_gt, modal2qp_stencil, modal2bd_stencil, modal2nodal_gt, modal2nodal_stencil
from numerical_flux import flux_bd_gt, compute_flux_gt, integrate_numerical_flux_stencil, flux_bd_stencil

from matmul.matmul_4_4 import matmul_4_4
from gt4py_config import dtype, backend, backend_opts

# @gtscript.stencil(backend=backend, **backend_opts)
# def elemwise_mult(
#     a: gtscript.Field[(dtype, (4,))],
#     b: gtscript.Field[(dtype, (4,))],
#     out: gtscript.Field[(dtype, (4,))]
# ):
#     with computation(PARALLEL), interval(...):
#         out[0,0,0][0] = a[0,0,0][0] * b[0,0,0][0]
#         out[0,0,0][1] = a[0,0,0][1] * b[0,0,0][1]
#         out[0,0,0][2] = a[0,0,0][2] * b[0,0,0][2]
#         out[0,0,0][3] = a[0,0,0][3] * b[0,0,0][3]

@gtscript.stencil(backend=backend)
def copy(
    in_phi: gtscript.Field[(dtype, (4,))],
    out_phi: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        out_phi[0,0,0][0] = in_phi[0, 0, 0][0]
        out_phi[0,0,0][1] = in_phi[0, 0, 0][1]
        out_phi[0,0,0][2] = in_phi[0, 0, 0][2]
        out_phi[0,0,0][3] = in_phi[0, 0, 0][3]

@gtscript.stencil(backend=backend, **backend_opts)
def inv_mass_stencil(
    inv_mass: gtscript.Field[(dtype, (4,4))],
    rhs: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        a_0, a_1, a_2, a_3 = matmul_4_4(inv_mass, rhs)
        rhs[0,0,0][0] = a_0
        rhs[0,0,0][1] = a_1
        rhs[0,0,0][2] = a_2
        rhs[0,0,0][3] = a_3


def integrate_flux(rhs, w, fx, fy, vander, determ, bd_det_x, bd_det_y):
    phi_grad_x = vander.grad_phi_x_gt
    phi_grad_y = vander.grad_phi_y_gt
    integrate_flux_stencil(w, fx, fy, phi_grad_x, phi_grad_y, rhs, determ, bd_det_x, bd_det_y)

def integrate_numerical_flux(rhs, w, f_n, f_s, f_e, f_w, vander, bd_det_x, bd_det_y):
    phi_n = vander.phi_bd_N_gt
    phi_s = vander.phi_bd_S_gt
    phi_e = vander.phi_bd_E_gt
    phi_w = vander.phi_bd_W_gt
    integrate_numerical_flux_stencil(w, f_n, f_s, f_e, f_w, phi_n, phi_s, phi_e, phi_w, rhs, bd_det_x, bd_det_y)

@gtscript.stencil(backend=backend, **backend_opts)
def subtract_boundary_term_stencil(
    rhs: gtscript.Field[(dtype, (4,))],
    boundary_term: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        rhs[0,0,0][0] -= boundary_term[0,0,0][0]
        rhs[0,0,0][1] -= boundary_term[0,0,0][1]
        rhs[0,0,0][2] -= boundary_term[0,0,0][2]
        rhs[0,0,0][3] -= boundary_term[0,0,0][3]

@gtscript.stencil(backend=backend, **backend_opts)
def runge_kuta_stencil(
    u_modal: gtscript.Field[(dtype, (4,))],
    rhs: gtscript.Field[(dtype, (4,))],
    dt: float
):
    with computation(PARALLEL), interval(...):
        u_modal[0,0,0][0] += dt * rhs[0,0,0][0]
        u_modal[0,0,0][1] += dt * rhs[0,0,0][1]
        u_modal[0,0,0][2] += dt * rhs[0,0,0][2]
        u_modal[0,0,0][3] += dt * rhs[0,0,0][3]

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
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,)))
    u_nodal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (n_qp2d,))) # for plotting

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
        # modal2qp_stencil(vander.phi_gt, uM_gt, u_qp)
        # flux_function_stencil(u_qp, fx, fy)
        # integrate_flux(rhs, wts2d, fx, fy, vander, determ, bd_det_x, bd_det_y)
        complete_flux_stencil(uM_gt, vander.phi_gt, vander.grad_phi_x_gt,
            vander.grad_phi_y_gt, wts2d, fx, rhs, determ, bd_det_x, bd_det_y
        )


        modal2bd_gt(vander.phi_bd_N_gt, uM_gt, u_n)
        modal2bd_gt(vander.phi_bd_S_gt, uM_gt, u_s)
        modal2bd_gt(vander.phi_bd_E_gt, uM_gt, u_e)
        modal2bd_gt(vander.phi_bd_W_gt, uM_gt, u_w)


        flux_bd_stencil(u_n, f_n, tmp, origin=(0,0,0), domain=(nx+2, nx+2,1))
        flux_bd_stencil(u_s, f_s, tmp, origin=(0,0,0), domain=(nx+2, ny+2, 1))
        flux_bd_stencil(u_e, tmp, f_e, origin=(0,0,0), domain=(nx+2, ny+2, 1))
        flux_bd_stencil(u_w, tmp, f_w, origin=(0,0,0), domain=(nx+2, ny+2, 1))

        compute_flux_gt(u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w, flux_n, flux_s, flux_e, flux_w)

        # print(f'{f_n = }\n\n')
        # print(f'{f_s = }\n\n')

        # print(f'{u_n = }\n\n')
        # print(f'{u_s = }\n\n')
        
        # print(f'{flux_n = }\n\n')

        integrate_numerical_flux(rhs, wts1d, flux_n, flux_s, flux_w, flux_e, vander, bd_det_x, bd_det_y)

        inv_mass_stencil(inv_mass, rhs) 
        runge_kuta_stencil(uM_gt, rhs, dt)
        # print(f'Iteration {i} done')
        
        # if i % plot_freq == 0:
        #     modal2nodal_stencil(vander.vander_gt, uM_gt, u_nodal)
        #     plotter.plot_solution(u_nodal, init=False, plot_type=plot_type)

    loop_end = time.perf_counter()

    print('--- Timings ---')
    print(f'Loop: {loop_end - loop_start}s')
    print(f'Allocation: {alloc_end - alloc_start}s')



