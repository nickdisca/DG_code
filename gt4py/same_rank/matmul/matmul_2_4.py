import gt4py.gtscript as gtscript

@gtscript.function
def matmul_2_4(matrix, vec):
	a_0 = matrix[0,0,0][0,0] * vec[0,0,0][0]+ matrix[0,0,0][0,1] * vec[0,0,0][1]+ matrix[0,0,0][0,2] * vec[0,0,0][2]+ matrix[0,0,0][0,3] * vec[0,0,0][3]
	a_1 = matrix[0,0,0][1,0] * vec[0,0,0][0]+ matrix[0,0,0][1,1] * vec[0,0,0][1]+ matrix[0,0,0][1,2] * vec[0,0,0][2]+ matrix[0,0,0][1,3] * vec[0,0,0][3]
	return a_0, a_1