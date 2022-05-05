import numpy as np

backend = "numpy"

# backend ="gt:cpu_ifirst" 
# backend ="gt:cpu_kfirst" 

# backend = "dace:cpu"

# backend = "cuda"


dtype = np.float64
backend_opts = {
    "rebuild": False
}

# dims
r = 2
dim=(r+1)**2
n_qp_1D = 4
n_qp = n_qp_1D**2

# dtypes
dtype_modal2qp_matrix = (dtype, (n_qp, dim))
dtype_modal2bd_matrix = (dtype, (n_qp_1D, dim))

dtype_qp_vec = (dtype, (n_qp,))
dtype_bd_vec = (dtype, (n_qp_1D,))
dtype_modal_vec = (dtype, (dim,))