import gt4py.gtscript as gtscript

@gtscript.function
def matmul(matrix, vec):
	a_0 = matrix[0,0] * vec[0]+ matrix[0,1] * vec[1]+ matrix[0,2] * vec[2]+ matrix[0,3] * vec[3]+ matrix[0,4] * vec[4]+ matrix[0,5] * vec[5]+ matrix[0,6] * vec[6]+ matrix[0,7] * vec[7]
	a_1 = matrix[1,0] * vec[0]+ matrix[1,1] * vec[1]+ matrix[1,2] * vec[2]+ matrix[1,3] * vec[3]+ matrix[1,4] * vec[4]+ matrix[1,5] * vec[5]+ matrix[1,6] * vec[6]+ matrix[1,7] * vec[7]
	a_2 = matrix[2,0] * vec[0]+ matrix[2,1] * vec[1]+ matrix[2,2] * vec[2]+ matrix[2,3] * vec[3]+ matrix[2,4] * vec[4]+ matrix[2,5] * vec[5]+ matrix[2,6] * vec[6]+ matrix[2,7] * vec[7]
	a_3 = matrix[3,0] * vec[0]+ matrix[3,1] * vec[1]+ matrix[3,2] * vec[2]+ matrix[3,3] * vec[3]+ matrix[3,4] * vec[4]+ matrix[3,5] * vec[5]+ matrix[3,6] * vec[6]+ matrix[3,7] * vec[7]
	a_4 = matrix[4,0] * vec[0]+ matrix[4,1] * vec[1]+ matrix[4,2] * vec[2]+ matrix[4,3] * vec[3]+ matrix[4,4] * vec[4]+ matrix[4,5] * vec[5]+ matrix[4,6] * vec[6]+ matrix[4,7] * vec[7]
	a_5 = matrix[5,0] * vec[0]+ matrix[5,1] * vec[1]+ matrix[5,2] * vec[2]+ matrix[5,3] * vec[3]+ matrix[5,4] * vec[4]+ matrix[5,5] * vec[5]+ matrix[5,6] * vec[6]+ matrix[5,7] * vec[7]
	a_6 = matrix[6,0] * vec[0]+ matrix[6,1] * vec[1]+ matrix[6,2] * vec[2]+ matrix[6,3] * vec[3]+ matrix[6,4] * vec[4]+ matrix[6,5] * vec[5]+ matrix[6,6] * vec[6]+ matrix[6,7] * vec[7]
	a_7 = matrix[7,0] * vec[0]+ matrix[7,1] * vec[1]+ matrix[7,2] * vec[2]+ matrix[7,3] * vec[3]+ matrix[7,4] * vec[4]+ matrix[7,5] * vec[5]+ matrix[7,6] * vec[6]+ matrix[7,7] * vec[7]
	a_8 = matrix[8,0] * vec[0]+ matrix[8,1] * vec[1]+ matrix[8,2] * vec[2]+ matrix[8,3] * vec[3]+ matrix[8,4] * vec[4]+ matrix[8,5] * vec[5]+ matrix[8,6] * vec[6]+ matrix[8,7] * vec[7]
	a_9 = matrix[9,0] * vec[0]+ matrix[9,1] * vec[1]+ matrix[9,2] * vec[2]+ matrix[9,3] * vec[3]+ matrix[9,4] * vec[4]+ matrix[9,5] * vec[5]+ matrix[9,6] * vec[6]+ matrix[9,7] * vec[7]
	a_10 = matrix[10,0] * vec[0]+ matrix[10,1] * vec[1]+ matrix[10,2] * vec[2]+ matrix[10,3] * vec[3]+ matrix[10,4] * vec[4]+ matrix[10,5] * vec[5]+ matrix[10,6] * vec[6]+ matrix[10,7] * vec[7]
	a_11 = matrix[11,0] * vec[0]+ matrix[11,1] * vec[1]+ matrix[11,2] * vec[2]+ matrix[11,3] * vec[3]+ matrix[11,4] * vec[4]+ matrix[11,5] * vec[5]+ matrix[11,6] * vec[6]+ matrix[11,7] * vec[7]
	a_12 = matrix[12,0] * vec[0]+ matrix[12,1] * vec[1]+ matrix[12,2] * vec[2]+ matrix[12,3] * vec[3]+ matrix[12,4] * vec[4]+ matrix[12,5] * vec[5]+ matrix[12,6] * vec[6]+ matrix[12,7] * vec[7]
	return a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12