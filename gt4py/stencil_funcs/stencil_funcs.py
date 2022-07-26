import gt4py.gtscript as gtscript
from gt4py_config import dtype, backend, backend_opts

@gtscript.stencil(backend=backend, **backend_opts)
def modal2qp(
	matrix: gtscript.Field[(dtype, (3,2))],
	in_vec: gtscript.Field[(dtype, (2,))],
	out_vec: gtscript.Field[(dtype, (3,))]
):
	out_vec[0,0,0][0] = matrix[0,0,0][0,0]*in_vec[0,0,0][0] + matrix[0,0,0][0,1]*in_vec[0,0,0][1]
	out_vec[0,0,0][1] = matrix[0,0,0][1,0]*in_vec[0,0,0][0] + matrix[0,0,0][1,1]*in_vec[0,0,0][1]
	out_vec[0,0,0][2] = matrix[0,0,0][2,0]*in_vec[0,0,0][0] + matrix[0,0,0][2,1]*in_vec[0,0,0][1]

@gtscript.stencil(backend=backend, **backend_opts)
def modal2qp_T(
	matrix: gtscript.Field[(dtype, (3,2))],
	in_vec: gtscript.Field[(dtype, (3,))],
	out_vec: gtscript.Field[(dtype, (2,))]
):
	out_vec[0,0,0][0] = matrix[0,0,0][0,0]*in_vec[0,0,0][0] + matrix[0,0,0][1,0]*in_vec[0,0,0][1] + matrix[0,0,0][2,0]*in_vec[0,0,0][2]
	out_vec[0,0,0][1] = matrix[0,0,0][0,1]*in_vec[0,0,0][0] + matrix[0,0,0][1,1]*in_vec[0,0,0][1] + matrix[0,0,0][2,1]*in_vec[0,0,0][2]

@gtscript.stencil(backend=backend, **backend_opts)
def elemwise_mult(
	vec_1: gtscript.Field[(dtype, (3,))],
	vec_2: gtscript.Field[(dtype, (3,))],
	out_vec: gtscript.Field[(dtype, (3,))]
):
	with computation(PARALLEL), interval(...):
		out_vec[0,0,0][0] = vec_1[0,0,0][0] * vec_2[0,0,0][0]
		out_vec[0,0,0][1] = vec_1[0,0,0][1] * vec_2[0,0,0][1]
		out_vec[0,0,0][2] = vec_1[0,0,0][2] * vec_2[0,0,0][2]