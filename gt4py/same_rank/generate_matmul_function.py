def generate_matmul_function(dim_x, dim_y):
    with open(f'matmul/matmul_{dim_x}_{dim_y}.py', 'w') as f:
        f.write('import gt4py.gtscript as gtscript\n\n')
        f.write('@gtscript.function\n')
        f.write('def matmul(matrix, vec):')
        alph = []
        for i in range(dim_x):
            alph.append(f'a_{i}')

        for i in range(dim_x):
            f.write(f'\n\t{alph[i]} = matrix[{i},0] * vec[0]')
            for j in range(1, dim_y):
                f.write(f'+ matrix[{i},{j}] * vec[{j}]')

        f.write(f'\n\treturn {alph[0]}')
        for i in range(1, dim_x):
            f.write(f', {alph[i]}')

if __name__ == "__main__":
    dim_x = 21; dim_y = 7
    generate_matmul_function(dim_x, dim_y)