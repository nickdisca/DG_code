import gt4py.gtscript as gtscript

@gtscript.function
def matmul_2_3_T(matrix, vec):
	a_0 = matrix[0,0] * vec[0]+ matrix[1,0] * vec[1]
	a_1 = matrix[0,1] * vec[0]+ matrix[1,1] * vec[1]
	a_2 = matrix[0,2] * vec[0]+ matrix[1,2] * vec[1]
	return a_0, a_1, a_2