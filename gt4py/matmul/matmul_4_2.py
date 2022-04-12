import gt4py.gtscript as gtscript

@gtscript.function
def matmul(matrix, vec):
	a_0 = matrix[0,0] * vec[0]+ matrix[0,1] * vec[1]
	a_1 = matrix[1,0] * vec[0]+ matrix[1,1] * vec[1]
	a_2 = matrix[2,0] * vec[0]+ matrix[2,1] * vec[1]
	a_3 = matrix[3,0] * vec[0]+ matrix[3,1] * vec[1]
	return a_0, a_1, a_2, a_3