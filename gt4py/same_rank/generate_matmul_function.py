def generate_matmul_function(dim_x, dim_y, transposed=False):
    if transposed:
        with open(f'matmul/matmul_{dim_x}_{dim_y}_T.py', 'w') as f:
            f.write('import gt4py.gtscript as gtscript\n\n')
            f.write('@gtscript.function\n')
            f.write(f'def matmul_{dim_x}_{dim_y}_T(matrix, vec):')
            alph = []
            for i in range(dim_x):
                alph.append(f'a_{i}')

            for i in range(dim_x):
                f.write(f'\n\t{alph[i]} = matrix[0,0,0][0,{i}] * vec[0,0,0][0]')
                for j in range(1, dim_y):
                    f.write(f'+ matrix[0,0,0][{j},{i}] * vec[0,0,0][{j}]')

            f.write(f'\n\treturn {alph[0]}')
            for i in range(1, dim_x):
                f.write(f', {alph[i]}')
    else:
        with open(f'matmul/matmul_{dim_x}_{dim_y}.py', 'w') as f:
            f.write('import gt4py.gtscript as gtscript\n\n')
            f.write('@gtscript.function\n')
            f.write(f'def matmul_{dim_x}_{dim_y}(matrix, vec):')
            alph = []
            for i in range(dim_x):
                alph.append(f'a_{i}')

            for i in range(dim_x):
                f.write(f'\n\t{alph[i]} = matrix[0,0,0][{i},0] * vec[0,0,0][0]')
                for j in range(1, dim_y):
                    f.write(f'+ matrix[0,0,0][{i},{j}] * vec[0,0,0][{j}]')

            f.write(f'\n\treturn {alph[0]}')
            for i in range(1, dim_x):
                f.write(f', {alph[i]}')

if __name__ == "__main__":
    dim_x = 2; dim_y = 4
    generate_matmul_function(dim_x, dim_y)
    generate_matmul_function(dim_x, dim_y, transposed=True)