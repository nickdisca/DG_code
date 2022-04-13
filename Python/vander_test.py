# %%
import numpy as np
# %%

n = 5
x = np.linspace(0, 1, n)

# -- 1D --
z1d = np.random.rand(n) # nodal values
vander1d = np.polynomial.legendre.legvander(x, n-1)
a = np.linalg.solve(vander1d, z1d) # modal values
print(z1d)
print(np.dot(vander1d, a))
# %%

# -- 2d --

x_tp = np.kron(x, np.ones(n))
y_tp = np.kron(np.ones(n), x)
z2d = np.random.rand(n*n)
vander2d = np.polynomial.legendre.legvander2d(x_tp, y_tp, [n-1, n-1])
a2d = np.linalg.solve(vander2d, z2d)
print(z2d)
print(np.dot(vander2d, a2d))
# %%
