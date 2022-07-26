import stencils

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
            "phi_bd_N": (0,), "phi_bd_S": (0,), "phi_bd_E": (0,), "phi_bd_W": (0,),
            "rhs_h": (0,0,0), "rhs_hu": (0,0,0), "rhs_hv": (0,0,0), "tmp": (0,0,0), "w": (0,),
            "inv_mass": (0,0,0), "cos_n": (0,), "cos_s": (0,)
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