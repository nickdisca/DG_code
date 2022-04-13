from element import Element
from functions import degree_distribution
from configuration import Config


import numpy as np

class Grid:
    def __init__(self, a, b, c, d, radius, dx, dy, u0_modal, matrices, rdist_type, plotter, r_max=Config.r_max, eq_type=Config.eq_type):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.hx = (b-a)/dx
        self.hy = (d-c)/dy
        self.matrices = matrices
        self.rdist = degree_distribution(rdist_type, dx, dy, r_max)
        self.r_max = r_max
        self.eq_type = eq_type
        self.x_c=np.linspace(a+self.hx/2,b-self.hx/2,dx); # Cell centers in X
        self.y_c=np.linspace(c+self.hy/2,d-self.hy/2,dy); # Cell centers in Y

        self.plotter = plotter

        self.elements = [[Element(i, j, self.x_c[i], self.y_c[j], self.hx, self.hy, matrices.pts2d_x, matrices.pts2d_y, self.rdist[i,j], matrices.phi_val_cell[self.rdist[i,j]], matrices.phi_grad_cell_x[self.rdist[i,j]], matrices.phi_grad_cell_y[self.rdist[i,j]], matrices.phi_val_bd_cell_n[self.rdist[i,j]], matrices.phi_val_bd_cell_s[self.rdist[i,j]], matrices.phi_val_bd_cell_e[self.rdist[i,j]], matrices.phi_val_bd_cell_w[self.rdist[i,j]], matrices.wts2d, eq_type) for j in range(dy)] for i in range(dx)]

        self.set_initial_conditions(u0_modal)


    def get_element(self, i, j):
        return self.elements[i][j]


    def set_initial_conditions(self, u0_modal):
        for i in range(self.dx):
            for j in range(self.dy):
                self.elements[i][j].set_initial_conditions(self.eq_type, u0_modal)

    def plot_solution(self):
        pass



