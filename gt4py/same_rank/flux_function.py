import gt4py.gtscript as gtscript
import gt4py as gt
from matmul.matmul_4_4_T import matmul_4_4_T
import numpy as np

from gt4py_config import dtype, backend, backend_opts

from matmul.matmul_4_4 import matmul_4_4

@gtscript.function
def flux_function(u_qp):
    fx_0 = u_qp[0,0,0][0]
    fx_1 = u_qp[0,0,0][1]
    fx_2 = u_qp[0,0,0][2]
    fx_3 = u_qp[0,0,0][3]

    fy_0 = u_qp[0,0,0][0]
    fy_1 = u_qp[0,0,0][1]
    fy_2 = u_qp[0,0,0][2]
    fy_3 = u_qp[0,0,0][3]
    return fx_0, fx_1, fx_2, fx_3, fy_0, fy_1, fy_2, fy_3

@gtscript.stencil(backend=backend, **backend_opts)
def flux_function_stencil(
    u: gtscript.Field[(dtype, (4,))],
    fx: gtscript.Field[(dtype, (4,))],
    fy: gtscript.Field[(dtype, (4,))]
):
    # with computation(PARALLEL), interval(...):
    #     fx_0, fx_1, fx_2, fx_3, fy_0, fy_1, fy_2, fy_3 = flux_function(u)
    #     fx[0,0,0][0] = fx_0
    #     fx[0,0,0][1] = fx_1
    #     fx[0,0,0][2] = fx_2
    #     fx[0,0,0][3] = fx_3

    #     fy[0,0,0][0] = fy_0
    #     fy[0,0,0][1] = fy_1
    #     fy[0,0,0][2] = fy_2
    #     fy[0,0,0][3] = fy_3

    with computation(PARALLEL), interval(...):
        fx[0,0,0][0] = u[0,0,0][0]
        fx[0,0,0][1] = u[0,0,0][1]
        fx[0,0,0][2] = u[0,0,0][2]
        fx[0,0,0][3] = u[0,0,0][3]

        fy[0,0,0][0] = u[0,0,0][0]
        fy[0,0,0][1] = u[0,0,0][1]
        fy[0,0,0][2] = u[0,0,0][2]
        fy[0,0,0][3] = u[0,0,0][3]

def flux_function_gt(u):
    nx, ny, nz, vec = u.shape
    fx = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    fy = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    flux_function_stencil(u, fx, fy)
    return fx, fy


@gtscript.stencil(backend=backend, **backend_opts)
def integrate_flux_stencil(
    w: gtscript.Field[(dtype, (4,))],
    fx: gtscript.Field[(dtype, (4,))],
    fy: gtscript.Field[(dtype, (4,))],
    phi_grad_x: gtscript.Field[(dtype, (4,4))],
    phi_grad_y: gtscript.Field[(dtype, (4,4))],
    rhs: gtscript.Field[(dtype, (4,))],
    determ: float,
    bd_det_x: float,
    bd_det_y: float

):
    with computation(PARALLEL), interval(...):
        fx[0,0,0][0] *= w[0,0,0][0]
        fx[0,0,0][1] *= w[0,0,0][1]
        fx[0,0,0][2] *= w[0,0,0][2]
        fx[0,0,0][3] *= w[0,0,0][3]

        x_0, x_1, x_2, x_3 = matmul_4_4_T(phi_grad_x, fx)

        fy[0,0,0][0] *= w[0,0,0][0]
        fy[0,0,0][1] *= w[0,0,0][1]
        fy[0,0,0][2] *= w[0,0,0][2]
        fy[0,0,0][3] *= w[0,0,0][3]

        y_0, y_1, y_2, y_3 = matmul_4_4_T(phi_grad_y, fy)

        rhs[0,0,0][0] = (x_0 / bd_det_x + y_0 / bd_det_y) * determ
        rhs[0,0,0][1] = (x_1 / bd_det_x + y_1 / bd_det_y) * determ
        rhs[0,0,0][2] = (x_2 / bd_det_x + y_2 / bd_det_y) * determ
        rhs[0,0,0][3] = (x_3 / bd_det_x + y_3 / bd_det_y) * determ



@gtscript.stencil(backend=backend, **backend_opts)
def complete_flux_stencil(
    u_modal: gtscript.Field[(dtype, (4,))],
    phi: gtscript.Field[(dtype, (4, 4))],
    phi_grad_x: gtscript.Field[(dtype, (4, 4))],
    phi_grad_y: gtscript.Field[(dtype, (4, 4))],
    w: gtscript.Field[(dtype, (4,))],
    f: gtscript.Field[(dtype, (4,))],
    rhs: gtscript.Field[(dtype, (4,))],
    determ: float,
    bd_det_x: float,
    bd_det_y: float
):
    with computation(PARALLEL), interval(...):
        # modal -> qp mapping
        a_0, a_1, a_2, a_3 = matmul_4_4(phi, u_modal)
        # in this case fx = fy = f
        f[0,0,0][0] = a_0 * w[0,0,0][0]
        f[0,0,0][1] = a_1 * w[0,0,0][1]
        f[0,0,0][2] = a_2 * w[0,0,0][2]
        f[0,0,0][3] = a_3 * w[0,0,0][3]

        x_0, x_1, x_2, x_3 = matmul_4_4_T(phi_grad_x, f)
        y_0, y_1, y_2, y_3 = matmul_4_4_T(phi_grad_y, f)

        rhs[0,0,0][0] = (x_0 / bd_det_x + y_0 / bd_det_y) * determ
        rhs[0,0,0][1] = (x_1 / bd_det_x + y_1 / bd_det_y) * determ
        rhs[0,0,0][2] = (x_2 / bd_det_x + y_2 / bd_det_y) * determ
        rhs[0,0,0][3] = (x_3 / bd_det_x + y_3 / bd_det_y) * determ


