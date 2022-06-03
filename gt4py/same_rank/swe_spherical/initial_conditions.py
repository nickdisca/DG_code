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

    if eq_type == "linear":
        neq = 1
        u0 = np.zeros((nx, ny, nz, dim))
        h0 = 0; h1 = 1; R = (b-a)/2/5; domaine_x_c = (a+b)/2; domaine_y_c=(c+d)/2
        u0_fun=lambda x, y:  h0+0.5*h1*(1+np.cos(np.pi*np.sqrt(np.square(x - domaine_x_c)+np.square(y - domaine_y_c))/R))*((np.sqrt(np.square(x-domaine_x_c)+np.square(y-domaine_y_c)))<R)

        # u0_fun=lambda x, y: np.ones(unif2d_x.shape) # constant state
        for i in range(nx):
            for j in range(ny):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y
                u0[i,j,0,:] = u0_fun(local_pos_x,local_pos_y)

    if eq_type == "swe":
        neq = 3
        h = np.zeros((nx, ny, nz, dim))
        u = np.zeros((nx, ny, nz, dim))
        v = np.zeros((nx, ny, nz, dim))

        h0=1000; h1=100; Cx=a+(b-a)*1/2; Cy=c+(d-c)*1/2; sigma=(a+b)/20
        h0_fun = lambda x, y:  h0 + h1 * np.exp(- ((x - Cx)**2 + (y - Cy)**2) / (2 * sigma**2))
        # h0_fun = lambda x, y:  h0

        for i in range(nx):
            for j in range(ny):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y
                h[i,j,0,:] = h0_fun(local_pos_x,local_pos_y)

        u0 = (h, u, v)
    if eq_type == "swe_sphere":
        neq = 3
        h = np.zeros((nx, ny, nz, dim))
        u = np.zeros((nx, ny, nz, dim))
        v = np.zeros((nx, ny, nz, dim))

        coriolis = np.zeros((nx, ny, nz, dim))

        #  === Case 2 Williamson ===
        g = 9.80616; h0 = 2.94e4/g; Omega=7.292e-5; uu0=2*np.pi*radius/(12*86400); angle=0
        h0_fun =lambda lam, th: h0-1/g*(radius*Omega*uu0+uu0**2/2)*(np.sin(th)*np.cos(angle)-np.cos(lam)*np.cos(th)*np.sin(angle))**2
        u_fun=lambda lam ,th: uu0*(np.cos(th)*np.cos(angle)+np.sin(th)*np.cos(lam)*np.sin(angle))
        v_fun=lambda lam, th: -uu0*np.sin(angle)*np.sin(lam)

        coriolis_fun = lambda lam, th: 2 * Omega * (np.sin(th)*np.cos(angle) - np.cos(th)*np.cos(lam)*np.sin(angle))
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