import numpy as np

def set_initial_conditions(x_c, y_c, a, b, c, d, dim, vander, eq_type="linear"):
    nx = x_c.shape[0]
    ny = y_c.shape[0]
    nz = 1

    hx = (b - a) / nx
    hy = (d - c) / ny

    unif2d_x = vander.unif2d_x
    unif2d_y = vander.unif2d_y

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
    return neq, u0