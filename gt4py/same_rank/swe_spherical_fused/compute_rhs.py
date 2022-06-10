import stencils
import boundary_conditions
import numpy as np

def compute_rhs(
    cons_var, rhs, cons_qp, fx, fy, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w,
    flux_n, flux_s, flux_e, flux_w, 
    determ, bd_det_x, bd_det_y,
    vander, inv_mass, cos_fact, sin_fact, cos_bd, coriolis, tmp,
    wts2d, wts1d, nx, ny, alpha, radius
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

        cos_n, cos_s, cos_e, cos_w = cos_bd

        # --- Boundary Integral ---
        origins = {
            "_all_": (0,0,0),'h_n': (1,1,0), 'h_s': (1,1,0), 'h_e': (1,1,0), 'h_w': (1,1,0),
            'hu_n': (1,1,0), 'hu_s': (1,1,0), 'hu_e': (1,1,0), 'hu_w': (1,1,0),
            'hv_n': (1,1,0), 'hv_s': (1,1,0), 'hv_e': (1,1,0), 'hv_w': (1,1,0)
        }

        stencils.first_stencil(
            vander.phi_gt, h, hu, hv, h_qp, hu_qp, hv_qp, 
            fh_x, fh_y, fhu_x, fhu_y, fhv_x, fhv_y,
            vander.grad_phi_x_gt, vander.grad_phi_y_gt, wts2d,
            rhs_h, rhs_hu, rhs_hv, cos_fact, g, determ, bd_det_x, bd_det_y,

            vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
            vander.phi_bd_W_gt, h_n, h_s, h_e, h_w, h,
            hu_n, hu_s, hu_e, hu_w, h,
            hv_n, hv_s, hv_e, hv_w, hv,
            origin=origins, domain=(nx,ny,1)
        )

        # # --- Flux Integral ---
        origins = {
            "_all_": (0,0,0),'u_n': (1,1,0), 'u_s': (1,1,0), 'u_e': (1,1,0), 'u_w': (1,1,0),
        }
        # stencils.flux_stencil_swe(
        #     vander.phi_gt, h, hu, hv, h_qp, hu_qp, hv_qp, 
        #     fh_x, fh_y, fhu_x, fhu_y, fhv_x, fhv_y,
        #     vander.grad_phi_x_gt, vander.grad_phi_y_gt, wts2d,
        #     rhs_h, rhs_hu, rhs_hv, cos_fact, g, determ, bd_det_x, bd_det_y
        # )

        # stencils.modal2bd(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, h_n, h_s, h_e, h_w, h,
        #     origin=origins, domain=(nx,ny,1)
        # )


        # stencils.modal2bd(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, hu_n, hu_s, hu_e, hu_w, hu,
        #     origin=origins, domain=(nx,ny,1)
        # )

        # stencils.modal2bd(
        #     vander.phi_bd_N_gt, vander.phi_bd_S_gt, vander.phi_bd_E_gt,
        #     vander.phi_bd_W_gt, hv_n, hv_s, hv_e, hv_w, hv,
        #     origin=origins, domain=(nx,ny,1)
        # )
        # --- END NOT FUSED ---

        boundary_conditions.apply_pbc(h_n)
        boundary_conditions.apply_pbc(h_s)
        boundary_conditions.apply_pbc(h_e)
        boundary_conditions.apply_pbc(h_w)

        boundary_conditions.apply_pbc(hu_n)
        boundary_conditions.apply_pbc(hu_s)
        boundary_conditions.apply_pbc(hu_e)
        boundary_conditions.apply_pbc(hu_w)

        boundary_conditions.apply_pbc(hv_n)
        boundary_conditions.apply_pbc(hv_s)
        boundary_conditions.apply_pbc(hv_e)
        boundary_conditions.apply_pbc(hv_w)

        stencils.flux_bd_stencil_swe(
            h_n, h_s, h_e, h_w, hu_n, hu_s, hu_e, hu_w,
            hv_n, hv_s, hv_e, hv_w, f_n_h, f_s_h, f_e_h, f_w_h,
            f_n_hu, f_s_hu, f_e_hu, f_w_hu, f_n_hv, f_s_hv, f_e_hv, f_w_hv, g,
            origin=(0,0,0), domain=(nx+2, ny+2, 1)
        )

        origins = {
            "_all_": (1,1,0),'flux_n': (0,0,0), 'flux_s': (0,0,0), 'flux_e': (0,0,0), 'flux_w': (0,0,0)
        }

        stencils.compute_num_flux(
            h_n, h_s, h_e, h_w, f_n_h, f_s_h, f_e_h, f_w_h,
            flux_n_h, flux_s_h, flux_e_h, flux_w_h, cos_n, cos_s, alpha,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.compute_num_flux(
            hu_n, hu_s, hu_e, hu_w, f_n_hu, f_s_hu, f_e_hu, f_w_hu,
            flux_n_hu, flux_s_hu, flux_e_hu, flux_w_hu, cos_n, cos_s, alpha,
            origin=origins, domain=(nx, ny, 1)
        )

        stencils.compute_num_flux(
            hv_n, hv_s, hv_e, hv_w, f_n_hv, f_s_hv, f_e_hv, f_w_hv,
            flux_n_hv, flux_s_hv, flux_e_hv, flux_w_hv, cos_n, cos_s, alpha,
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


        # --- Fused ---
        stencils.fused_source_coriolis_stencil(
            vander.phi_gt, inv_mass, h_qp, hu_qp, hv_qp, tmp, coriolis,
            cos_fact, sin_fact, rhs_h, rhs_hu, rhs_hv, wts2d, g, radius, determ
        )
        # --- NOT FUSED ---
        # stencils.source_stencil(
        #     vander.phi_gt, h_qp, sin_fact, rhs_hv, wts2d, g, determ
        # )

        # stencils.coriolis_stencil(
        #     vander.phi_gt, coriolis, hu_qp, hv_qp, cos_fact, rhs_hu, rhs_hv, wts2d, radius, determ
        # )

        # stencils.inv_mass_stencil(rhs_h, rhs_hu, rhs_hv, tmp, inv_mass, radius)

