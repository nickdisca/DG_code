import pytest
import sys
import matplotlib.pyplot as plt
sys.path.append('..')

from functions import compute_mass, degree_distribution, norm_coeffs, nodal2modal, initial_conditions
from grid import Grid
from matrices import Matrices
import numpy as np


# eq_type = 'linear'
eq_type = 'adv_sphere'
r_max = 2

radius=6.37122e6;
d1 = 20; d2 = 20;
a=0; b=2*np.pi; c=-np.pi/2; d=np.pi/2;
hx=(b-a)/d1; hy=(d-c)/d2;
x_c=np.linspace(a+hx/2,b-hx/2,d1); # Cell centers in X
y_c=np.linspace(c+hy/2,d-hy/2,d2); # Cell centers in Y
n_qp_1D=4

rdist = degree_distribution("unif",d1,d2,r_max);
[pts,wts]=np.polynomial.legendre.leggauss(n_qp_1D)
pts2d_x = np.kron(pts,np.ones(n_qp_1D))
pts2d_y = np.kron(np.ones(n_qp_1D),pts)
wts2d   = np.kron(wts,wts)

unif2d_x = {}
unif2d_y = {}
for r in range(r_max+1):
    unif=np.linspace(-1,1,r+1)
    unif2d_x[r] = np.kron(unif,np.ones(r+1))
    unif2d_y[r] = np.kron(np.ones(r+1),unif)

phi_val_cell={}
V = {}
for r in range(r_max+1):
    num_coeff = r+1
    coeffs = norm_coeffs(num_coeff)
    legvander2d = np.polynomial.legendre.legvander2d(unif2d_x[r],unif2d_y[r],[r,r])
    V[r] = np.multiply(legvander2d,coeffs)
    legvander2d = np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[r, r])
    phi_val_cell[r] = np.multiply(legvander2d,coeffs)

mass, _ = compute_mass(phi_val_cell, wts2d, d1, d2, rdist, hx, hy, y_c, pts2d_y, eq_type);

neq, u0 = initial_conditions(eq_type, a, b, c, d, radius, hx, hy, d1, d2, unif2d_x, unif2d_y, rdist)
matrices = Matrices(quad_type="leg", n_qp_1d=n_qp_1D,r_max=r_max)
u0_modal = nodal2modal(d1,d2,neq,u0,V,rdist)
grid = Grid(a, b, c, d, radius, d1, d2, u0_modal, matrices, "unif", r_max=r_max, eq_type=eq_type)


def test_phi():
    for j in range(d2):
        for i in range(d1):
            r_loc = rdist[i,j]
            matrix=phi_val_cell[r_loc]
            grid_matrix = grid.get_element(i, j).phi_cell
            # print(f"matrix = {matrix}")
            # print(f"grid_matrix = {grid_matrix}")
            assert matrix.shape == grid_matrix.shape
            np.testing.assert_equal(matrix, grid_matrix)

def test_mass():
    for j in range(d2):
        for i in range(d1):
            matrix = mass[i,j]
            grid_matrix = grid.get_element(i, j).mass
            # print(f"matrix = {matrix}")
            # print(f"grid_matrix = {grid_matrix}")
            assert matrix.shape == grid_matrix.shape
            np.testing.assert_equal(matrix, grid_matrix)


def test_centers():
    for j in range(d2):
        for i in range(d1):
            x_c_grid = grid.get_element(i, j).x_c
            y_c_grid = grid.get_element(i, j).y_c
            np.testing.assert_equal(y_c[j], y_c_grid)
            np.testing.assert_equal(x_c[i], x_c_grid)

def test_initial_conditions():
    for i in range(d1):
        for j in range(d2):
            u_grid = grid.get_element(i, j).u
            neq = u_grid.shape[1]
            for eq in range(neq):
                tmp_shape = u_grid[eq].shape
                np.testing.assert_array_equal(u_grid[:,eq], u0_modal[i,j,eq])
                # print(f'element: {u_grid[eq].reshape(-1)}')
                # print(f'initial: {u0_modal[i,j,eq]}')
    

def plot_initial_condition(x=True):
    # p = []
    p = np.zeros(0)
    if x:
        for i in range(d1):
            u_grid = grid.get_element(i, int(d2/2)).u0
            p = np.append(p, u_grid[0].reshape(-1))
    else:
        for j in range(d2):
            u_grid = grid.get_element(int(d1/2), j).u0
            p = np.append(p, u_grid[0].reshape(-1))

    plt.plot(p)
    plt.show()



if __name__ == '__main__':
    # test_initial_conditions()
    plot_initial_condition(x=False)