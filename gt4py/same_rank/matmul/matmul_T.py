@gtscript.stencil(
def matmul_T(matrix, in_vec, out_vec):
	out_vec[0,0,0][0] = matrix[0,0,0][0,0] * vec[0,0,0][0]+ matrix[0,0,0][1,0] * vec[0,0,0][1]
	out_vec[0,0,0][1] = matrix[0,0,0][0,1] * vec[0,0,0][0]+ matrix[0,0,0][1,1] * vec[0,0,0][1]
	out_vec[0,0,0][2] = matrix[0,0,0][0,2] * vec[0,0,0][0]+ matrix[0,0,0][1,2] * vec[0,0,0][1]
	out_vec[0,0,0][3] = matrix[0,0,0][0,3] * vec[0,0,0][0]+ matrix[0,0,0][1,3] * vec[0,0,0][1]