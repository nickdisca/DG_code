import numpy as np

def set_initial_conditions(x_c, y_c, a, b, c, d, dim, vander, eq_type="linear"):
    nx = x_c.shape[0]
    ny = y_c.shape[0]
    hx = (b - a) / nx
    hy = (d - c) / ny

    unif2d_x = vander.unif2d_x
    unif2d_y = vander.unif2d_y

    if eq_type == "linear":
        neq = 1
        u0 = np.zeros((nx, ny, neq, dim))
        # -- Cosine bell --
        h0 = 0; h1 = 1; R = (b-a)/10; domaine_x_c = (a+b)/2; domaine_y_c=(c+d)/2
        # u0_fun=lambda x, y:  h0+0.5*h1*(1+np.cos(np.pi*np.sqrt(np.square(x - domaine_x_c)+np.square(y - domaine_y_c))/R))*((np.sqrt(np.square(x-domaine_x_c)+np.square(y-domaine_y_c)))<R)

        # -- 2D sin ---
        u0_fun = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

        # --- Constant  ---
        # u0_fun = lambda x, y: 1

        # u0_fun=lambda x, y: np.ones(unif2d_x.shape) # constant state
        for i in range(nx):
            for j in range(ny):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y
                u0[i,j,0,:] = u0_fun(local_pos_x,local_pos_y)

    return neq, u0
