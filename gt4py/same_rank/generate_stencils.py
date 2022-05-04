def generate_matmul_stencil(rows, cols, func_name, transposed=False):
    if transposed:
        field_matrix = f"gtscript.Field[(dtype, ({cols},{rows}))]"
        in_field = f"gtscript.Field[(dtype, ({cols},))]"
        out_field = f"gtscript.Field[(dtype, ({rows},))]"
        with open(f'stencil_funcs/stencil_funcs.py', 'a') as f:
            f.write('@gtscript.stencil(backend=backend, **backend_opts)\n')
            f.write(f'def {func_name}_T(\n\tmatrix: {field_matrix},\n\tin_vec: {in_field},\n\tout_vec: {out_field}\n):')
            for j in range(rows):
                f.write(f'\n\tout_vec[0,0,0][{j}] = matrix[0,0,0][0,{j}]*in_vec[0,0,0][0]')
                for i in range(1, cols):
                    f.write(f' + matrix[0,0,0][{i},{j}]*in_vec[0,0,0][{i}]')

            f.write('\n\n')

    else:
        field_matrix = f"gtscript.Field[(dtype, ({cols},{rows}))]"
        in_field = f"gtscript.Field[(dtype, ({rows},))]"
        out_field = f"gtscript.Field[(dtype, ({cols},))]"
        with open(f'stencil_funcs/stencil_funcs.py', 'a') as f:
            f.write('@gtscript.stencil(backend=backend, **backend_opts)\n')
            f.write(f'def {func_name}(\n\tmatrix: {field_matrix},\n\tin_vec: {in_field},\n\tout_vec: {out_field}\n):')
            for i in range(cols):
                f.write(f'\n\tout_vec[0,0,0][{i}] = matrix[0,0,0][{i},0]*in_vec[0,0,0][0]')
                for j in range(1, rows):
                    f.write(f' + matrix[0,0,0][{i},{j}]*in_vec[0,0,0][{j}]')

            f.write('\n\n')


def generate_elemwise_mult(dim):
    field = f"gtscript.Field[(dtype, ({dim},))]"
    with open(f'stencil_funcs/stencil_funcs.py', 'a') as f:
        f.write('@gtscript.stencil(backend=backend, **backend_opts)\n')
        f.write(f'def elemwise_mult(\n\tvec_1: {field},\n\tvec_2: {field},\n\tout_vec: {field}\n):')
        f.write(f'\n\twith computation(PARALLEL), interval(...):')
        for i in range(dim):
            f.write(f'\n\t\tout_vec[0,0,0][{i}] = vec_1[0,0,0][{i}] * vec_2[0,0,0][{i}]')

def generate_imports():
    with open(f'stencil_funcs/stencil_funcs.py', 'w') as f:
        f.write('import gt4py.gtscript as gtscript\n')
        f.write('from gt4py_config import dtype, backend, backend_opts\n\n')


if __name__ == "__main__":

    rows = 2; cols = 3
    func_name = 'modal2qp'
    # generate_matmul_stencil(dim_x, dim_y)

    generate_imports()
    generate_matmul_stencil(rows, cols, func_name, transposed=False)
    generate_matmul_stencil(rows, cols, func_name, transposed=True)
    generate_elemwise_mult(cols)