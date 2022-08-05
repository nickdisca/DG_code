import numpy as np
import matplotlib.pyplot as plt

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

        self.fig, self.ax = plt.subplots()


    def plot_solution(self, u, z_comp=0, init=False, plot_type='contour',show=False, save=False):

        x_u    = np.zeros(self.nx*self.r)
        y_u    = np.zeros(self.ny*self.r)
        unif   = np.linspace(-1,1,self.r)
        for i in range(self.nx) :
            x_u[i*self.r:(i+1)*self.r] = self.x_c[i]+self.hx*unif/2
        for j in range(self.ny) :
            y_u[j*self.r:(j+1)*self.r] = self.y_c[j]+self.hy*unif/2

        Y, X = np.meshgrid(y_u, x_u)  # Transposed to visualize properly
        Z    = np.zeros((self.nx*self.r,self.nx*self.r))
        
        for k in range(self.neq):
            for j in range(self.ny):
                for i in range(self.nx):
                    Z[i*self.r:(i+1)*self.r,j*self.r:(j+1)*self.r] = u[i,j,z_comp,:].reshape(self.r,self.r)
        # Z[np.abs(Z) < np.amax(Z)/1000.0] = 0.0   # Clip all values less than 1/1000 of peak
                    
        if plot_type == 'contour':
            # self.fig.clear()
            CS = self.ax.contourf(X, Y, Z, cmap='jet')
            if init:
                self.cbar = self.fig.colorbar(CS)
            else:
                if hasattr(self, 'cbar'):
                    self.cbar.remove()
                self.cbar = self.fig.colorbar(CS)
        if show:
            plt.draw()
            plt.show()
        else:
            plt.pause(0.005)
        if save:
            plt.savefig("img/final_step.svg", dpi=150)
