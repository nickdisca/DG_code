import numpy as np

def set_initial_conditions(x_c, y_c, a, b, c, d, radius, dim, n_qp, vander, pts2d_x, pts2d_y, eq_type="linear"):
    nx = x_c.shape[0]
    ny = y_c.shape[0]
    nz = 1

    hx = (b - a) / nx
    hy = (d - c) / ny

    unif2d_x = vander.unif2d_x
    unif2d_y = vander.unif2d_y

    coriolis = None

    neq = 3
    h = np.zeros((nx, ny, nz, dim))
    u = np.zeros((nx, ny, nz, dim))
    v = np.zeros((nx, ny, nz, dim))

    coriolis = np.zeros((nx, ny, nz, dim))

    #  === Case 2 Williamson ===
    # g = 9.80616; h0 = 2.94e4/g; Omega=7.292e-5; uu0=2*np.pi*radius/(12*86400); angle=0
    # h0_fun =lambda lam, th: h0-1/g*(radius*Omega*uu0+uu0**2/2)*(np.sin(th)*np.cos(angle)-np.cos(lam)*np.cos(th)*np.sin(angle))**2
    # u_fun=lambda lam, th: uu0*(np.cos(th)*np.cos(angle)+np.sin(th)*np.cos(lam)*np.sin(angle))
    # v_fun=lambda lam, th: -uu0*np.sin(angle)*np.sin(lam)

    # coriolis_fun = lambda lam, th: 2 * Omega * (np.sin(th)*np.cos(angle) - np.cos(th)*np.cos(lam)*np.sin(angle))
    # === END ===

    # === Case 6 Williamson ===
    g=9.80616; Omega=7.292e-5; omega=7.848e-6; K=omega; h0=8e3; R=4
    A_fun = lambda lam, th: 0.5*omega * (2*Omega+omega) * np.cos(th)**2 + 0.25*K*K*np.cos(th)**(2*R) * ((R+1)*np.cos(th)**2 + (2*R*R - R - 2) - 2*R*R*np.cos(th)**(-2))
    B_fun = lambda lam, th: 2*(Omega+omega)*K*np.cos(th)**R * ((R*R + 2*R + 2) - (R+1)**2*np.cos(th)**2) / ((R+1) * (R+2))
    C_fun = lambda lam, th: 0.25*K*K*np.cos(th)**(2*R) * ((R+1)*np.cos(th)**2 - (R+2))
    
    h0_fun = lambda lam, th: h0 + radius*radius/g*(A_fun(lam,th) + B_fun(lam,th)*np.cos(R*lam) + C_fun(lam,th)*np.cos(2*R*lam))
    u_fun = lambda lam, th: radius*omega*np.cos(th) + radius*K*np.cos(th)**(R-1) * (R*np.sin(th)**2 - np.cos(th)**2) * (np.cos(R*lam))
    v_fun = lambda lam, th: -radius*K*R*np.cos(th)**(R-1)*np.sin(th)*np.sin(R*lam)

    coriolis_fun = lambda lam,th: 2*Omega*np.sin(th)
    # === END ===

    for i in range(nx):
        for j in range(ny):
            local_pos_x = x_c[i] + 0.5*hx*unif2d_x
            local_pos_y = y_c[j] + 0.5*hy*unif2d_y
            local_qp_x = x_c[i] + 0.5*hx*pts2d_x
            local_qp_y = y_c[j] + 0.5*hy*pts2d_y
            h[i,j,0,:] = h0_fun(local_pos_x,local_pos_y)
            u[i,j,0,:] = u_fun(local_pos_x,local_pos_y)
            v[i,j,0,:] = v_fun(local_pos_x,local_pos_y)
            coriolis[i,j,0,:] = coriolis_fun(local_qp_x, local_qp_y)

    u0 = (h, u, v)
    return neq, u0, coriolis