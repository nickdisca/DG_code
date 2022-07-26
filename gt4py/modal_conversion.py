import numpy as np
from matmul import matmul_16_9
from matmul.matmul_4_4 import matmul_4_4
from matmul.matmul_2_4 import matmul_2_4
from matmul.matmul_2_4_T import matmul_2_4_T
import gt4py.gtscript as gtscript
import gt4py as gt

from boundary_conditions import apply_pbc
from gt4py_config import dtype, backend, backend_opts

def nodal2modal_gt(vander, in_nodal):
    nx, ny, nz, _ = in_nodal.shape
    vec = vander.shape[3]
    out_modal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    nodal2modal_stencil(vander, in_nodal, out_modal)
    return out_modal

def modal2nodal_gt(vander, in_modal):
    nx, ny, nz, vec = in_modal.shape
    out_nodal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    modal2nodal_stencil(vander, in_modal, out_nodal)
    return out_nodal


@gtscript.stencil(backend=backend, **backend_opts)
def nodal2modal_stencil(
    vander: gtscript.Field[(dtype, (4, 4))],
    in_nodal: gtscript.Field[(dtype, (4,))],
    out_modal: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        a_0, a_1, a_2, a_3 = matmul_4_4(vander, in_nodal)
        out_modal[0,0,0][0] = a_0
        out_modal[0,0,0][1] = a_1
        out_modal[0,0,0][2] = a_2
        out_modal[0,0,0][3] = a_3

@gtscript.stencil(backend=backend, **backend_opts)
def modal2nodal_stencil(
    vander_inv: gtscript.Field[(dtype, (4, 4))],
    in_modal: gtscript.Field[(dtype, (4,))],
    out_nodal: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        a_0, a_1, a_2, a_3 = matmul_4_4(vander_inv, in_modal)
        out_nodal[0,0,0][0] = a_0
        out_nodal[0,0,0][1] = a_1
        out_nodal[0,0,0][2] = a_2
        out_nodal[0,0,0][3] = a_3

def modal2qp_gt(phi, in_modal):
    nx, ny, nz, _ = in_modal.shape
    vec = phi.shape[3]
    out_nodal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    modal2qp(phi, in_modal, out_nodal)
    return out_nodal

@gtscript.stencil(backend=backend, **backend_opts)
def modal2qp_stencil(
    phi: gtscript.Field[(dtype, (4, 4))],
    in_modal: gtscript.Field[(dtype, (4,))],
    out_qp: gtscript.Field[(dtype, (4,))]
):
    with computation(PARALLEL), interval(...):
        a_0, a_1, a_2, a_3 = matmul_4_4(phi, in_modal)
        out_qp[0,0,0][0] = a_0
        out_qp[0,0,0][1] = a_1
        out_qp[0,0,0][2] = a_2
        out_qp[0,0,0][3] = a_3


# def modal2bd_gt(phi, in_modal):
#     nx, ny, nz, _ = in_modal.shape
#     vec = phi.shape[3]
#     out = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
#         shape=(nx+2, ny+2, nz), dtype=(dtype, (vec,)))
#     origins = {"out": (1,1,0)}
#     # origins = {"phi": (0,0,0), "in_modal": (0,0,0), "out": (1,1,0)}
#     modal2bd_stencil(phi, in_modal, out, origin=origins, domain=(nx,ny,1))
#     # periodic boundary conditions
#     apply_pbc(out)
#     return out
def modal2bd_gt(phi, in_modal, out):
    nx, ny, nz, _ = in_modal.shape
    vec = phi.shape[3]
    origins = {"phi": (0,0,0), "in_modal": (0,0,0), "out": (1,1,0)}
    modal2bd_stencil(phi, in_modal, out, origin=origins, domain=(nx,ny,1))
    # periodic boundary conditions
    apply_pbc(out)

@gtscript.stencil(backend=backend, **backend_opts)
def modal2bd_stencil(
    phi_bd: gtscript.Field[(dtype, (2, 4))],
    in_nodal: gtscript.Field[(dtype, (4,))],
    out: gtscript.Field[(dtype, (2,))]
):
    with computation(PARALLEL), interval(...):
        a_0, a_1  = matmul_2_4(phi_bd, in_nodal)
        out[0,0,0][0] = a_0
        out[0,0,0][1] = a_1