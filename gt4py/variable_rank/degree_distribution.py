from math import floor
import numpy as np

import gt4py as gt

def degree_distribution(distribution, d1, d2, r_max, backend="gtc:numpy"):
# %compute spatial distribution of the degrees

    if distribution == 'unif':
        r = r_max * np.ones((d2,d1), dtype=np.int)

    elif distribution == 'y_dep':
        r_vec = np.round((r_max-1) / (floor(d2/2)-1) * (np.arange(0, floor(d2/2)))+1).astype(int)
        # r_vec = np.round((r_max-1) / (floor(d2/2)-1) * (np.arange(0, floor(d2/2)))+1)
        r_vec = np.concatenate((r_vec, np.flipud(r_vec)))
        if d2 % 2 == 1:
            mid = len(r_vec) // 2
            r_vec = np.insert(r_vec, [mid], r_vec[mid])
        
        r = np.tile(r_vec.reshape(-1, 1), d1)

    elif distribution == 'y_incr':
        r_vec = np.round((r_max-1) / (d2-1) * (np.arange(0, d2))+1).astype(int)
        r = np.tile(r_vec.reshape(-1, 1), d1)

    else:
        raise Exception(f"Degree distribution unknown {distribution=}")
    

    r[r>r_max] = r_max

    return gt.storage.from_array(r,  backend, default_origin=(0,0,0))


# if __name__ == "__main__":
#     d1 = 20; d2 = 20; r_max = 2; distribution = "y_incr"
#     degree_distribution(distribution, d1, d2, r_max)