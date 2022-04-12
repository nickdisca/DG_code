import numpy as np

def set_initial_condions(x_c, y_c, a, b, c, d, unif_x, unif_y, eq_type="linear", rdist=None):
    nx = x_c.shape[0]
    ny = y_c.shape[0]

    if eq_type == "linear":
        neq = 1
        u = np.zeros((nx, ny, neq))
        h0 = 0; h1 = 1; R = (b-a)/2/5; x_c = (a+b)/2; y_c=(c+d)/2
        u0_fun=lambda x_c,y_c:  h0+h1/2*(1+np.cos(np.pi*np.sqrt(np.square(x_c-x_c)+np.square(y_c-y_c)))/R)*(np.sqrt(np.square(x_c-x_c)+np.square(y_c-y_c))<R).astype(np.double)
        for i in range(nx):
            for j in range(ny):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x[rdist[i,j]]
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y[rdist[i,j]]
                u0[i,j,0] = u0_fun(local_pos_x,local_pos_y)