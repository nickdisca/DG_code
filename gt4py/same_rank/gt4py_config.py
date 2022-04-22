import numpy as np

# backend = "numpy"

backend ="gt:cpu_ifirst" 
# backend ="gt:cpu_kfirst" 

# backend = "dace:cpu"

# backend = "cuda"

dtype = np.float64
backend_opts = {
    "rebuild": False
}
