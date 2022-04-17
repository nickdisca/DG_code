import gt4py.gtscript as gtscript

@gtscript.function
def matmul_2_4_T(matrix, vec):
	a_0 = matrix[0,0,0][0,0] * vec[0,0,0][0]+ matrix[0,0,0][1,0] * vec[0,0,0][1]+ matrix[0,0,0][2,0] * vec[0,0,0][2]+ matrix[0,0,0][3,0] * vec[0,0,0][3]
	a_1 = matrix[0,0,0][0,1] * vec[0,0,0][0]+ matrix[0,0,0][1,1] * vec[0,0,0][1]+ matrix[0,0,0][2,1] * vec[0,0,0][2]+ matrix[0,0,0][3,1] * vec[0,0,0][3]
	return a_0, a_1