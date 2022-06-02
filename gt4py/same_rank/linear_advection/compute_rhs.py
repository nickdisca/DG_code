import stencils
import boundary_conditions

def compute_rhs(
    uM_gt, rhs, u_qp, fx, fy, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
    flux_n, flux_s, flux_e, flux_w, 
    determ, bd_det_x, bd_det_y,
    vander, inv_mass, wts2d, wts1d, nx, ny, dt, alpha
):
        stencils.flux_stencil(
            vander.phi_gt, uM_gt, vander.grad_phi_x_gt,
            vander.grad_phi_y_gt, wts2d, rhs, determ, bd_det_x, bd_det_y
        )

        # --- Boundary Integral ---
        # origins = {
        #     "_all_": (0,0,0),'u_n': (1,1,0), 'u_s': (1,1,0), 'u_e': (1,1,0), 'u_w': (1,1,0)
        # }
        stencils.modal2bd(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, u_n, u_s, u_e, u_w, uM_gt
        )
        # boundary_conditions.apply_pbc(u_n)
        # boundary_conditions.apply_pbc(u_s)
        # boundary_conditions.apply_pbc(u_e)
        # boundary_conditions.apply_pbc(u_w)

        stencils.flux_bd_stencil(
            u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w
        )

        # origins = {
        #     "_all_": (0,0,0), "f_n": (0,1,0), "f_s": (0,-1,0), "f_e": (-1,0,0), "f_w": (1,0,0),
        #     "u_n": (0,1,0), "u_s": (0,-1,0), "u_e": (-1,0,0), "u_w": (1,0,0)
        # }
        # origins = {
        #     "f_n": (1,1,0), "f_s": (1,1,0), "f_e": (1,1,0), "f_w": (1,1,0),
        #     "u_n": (1,1,0), "u_s": (1,1,0), "u_e": (1,1,0), "u_w": (1,1,0)
        # }
        # origins = {"_all_": (1,1,0)}
        stencils.compute_num_flux(
            u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
            flux_n, flux_s, flux_e, flux_w, alpha,
            origin=(1,1,0), domain=(nx-2, ny-2, 1)
        )

        stencils.compute_num_flux(
            u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
            flux_n, flux_s, flux_e, flux_w, alpha,
            origin=(1,1,0), domain=(nx-2, ny-2, 1)
        )

        stencils.integrate_num_flux(
            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, flux_n, flux_s, flux_e, flux_w, wts1d, rhs,
            inv_mass, bd_det_x, bd_det_y
        )