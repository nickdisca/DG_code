def apply_pbc(u):
    nx, ny, nz, _ = u.shape
    u[1:-1, 0] = u[1:-1, -2] # north
    u[1:-1, -1] = u[1:-1, 1] # south
    u[-1, 1:-1] = u[1, 1:-1] # east
    u[0, 1:-1] = u[-2, 1:-1] # west

    return u
