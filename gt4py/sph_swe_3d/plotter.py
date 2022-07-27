import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class Plotter():
    def __init__(self, x_c, y_c, r, nx, ny, neq, hx, hy, plot_freq, plot_type):
        self.x_c = x_c
        self.y_c = y_c
        self.r = r
        self.nx = nx
        self.ny = ny
        self.neq = neq
        self.hx = hx
        self.hy = hy
        self.plot_freq = plot_freq
        self.plot_type = plot_type

        self.fig, self.ax = plt.subplots(1, 3, figsize=(16, 8))
        self.cbar = [None] * 3


    def plot_solution(self,u,init=False, plot_type='contour', fname='',title='',show=False):
        self.fig.suptitle(title)
        z_component = 0
        x_u    = np.zeros(self.nx*self.r)
        y_u    = np.zeros(self.ny*self.r)
        unif   = np.linspace(-1,1,self.r)
        for i in range(self.nx) :
            x_u[i*self.r:(i+1)*self.r] = self.x_c[i]+self.hx*unif/2
        for j in range(self.ny) :
            y_u[j*self.r:(j+1)*self.r] = self.y_c[j]+self.hy*unif/2


        Y, X = np.meshgrid(y_u, x_u)  # Transposed to visualize properly
        for idx, cons_var in enumerate(u):
            if idx > 0:
                cons_var = cons_var / u[0]
            Z    = np.zeros((self.nx*self.r,self.nx*self.r))
            for k in range(self.neq):
                for j in range(self.ny):
                    for i in range(self.nx):
                        Z[i*self.r:(i+1)*self.r,j*self.r:(j+1)*self.r] = cons_var[i,j,z_component,:].reshape(self.r,self.r)
            # Z[np.abs(Z) < np.amax(Z)/1000.0] = 0.0   # Clip all values less than 1/1000 of peak
                        
            if plot_type == 'contour':
                # self.fig.clear()
                CS = self.ax[idx].contourf(X, Y, Z, cmap='jet')
                if init:
                    self.cbar[idx] = self.fig.colorbar(CS, ax=self.ax[idx])
                else:
                    if self.cbar[idx] is not None:
                        self.cbar[idx].remove()
                    self.cbar[idx] = self.fig.colorbar(CS, ax=self.ax[idx])

            if show:
                plt.draw()
                plt.show()
            if fname:
                plt.savefig('img/' + fname, dpi=300)
            else:
                plt.pause(0.005)
