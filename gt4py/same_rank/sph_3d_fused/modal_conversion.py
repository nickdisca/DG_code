import numpy as np
import gt4py.gtscript as gtscript
import gt4py as gt

from boundary_conditions import apply_pbc
from gt4py_config import dtype, backend, backend_opts, dim, n_qp

def nodal2modal_gt(vander, in_nodal, backend=backend):
    nx, ny, nz, _ = in_nodal.shape
    vec = vander.shape[3]
    out_modal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    nodal2modal_stencil(vander, in_nodal, out_modal)
    return out_modal

def modal2nodal_gt(vander, in_modal, backend=backend):
    nx, ny, nz, vec = in_modal.shape
    out_nodal = gt.storage.zeros(backend=backend, default_origin=(0,0,0),
        shape=(nx, ny, nz), dtype=(dtype, (vec,)))
    modal2nodal_stencil(vander, in_modal, out_nodal)
    return out_nodal


@gtscript.stencil(backend=backend, **backend_opts)
def nodal2modal_stencil(
    vander: gtscript.Field[(dtype, (dim, dim))],
    in_nodal: gtscript.Field[(dtype, (dim,))],
    out_modal: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out_modal = vander @ in_nodal

@gtscript.stencil(backend=backend, **backend_opts)
def modal2nodal_stencil(
    vander_inv: gtscript.Field[(dtype, (dim, dim))],
    in_modal: gtscript.Field[(dtype, (dim,))],
    out_nodal: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out_nodal = vander_inv @ in_modal

@gtscript.stencil(backend=backend, **backend_opts)
def integration(
    phi: gtscript.Field[(dtype, (n_qp, dim))],
    w: gtscript.Field[(dtype, (n_qp,))],
    in_modal: gtscript.Field[(dtype, (dim,))],
    determ: float,
    out_nodal: gtscript.Field[(dtype, (dim,))]
):
    with computation(PARALLEL), interval(...):
        out_nodal = (phi @ in_modal) * w * determ


