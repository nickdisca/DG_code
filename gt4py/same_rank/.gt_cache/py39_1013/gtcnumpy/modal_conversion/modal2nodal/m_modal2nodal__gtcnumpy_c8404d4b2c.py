import pathlib
import time

import numpy as np
from numpy import dtype
import sys
import pathlib
import numpy

path_backup = sys.path.copy()
sys.path.append(str(pathlib.Path(__file__).parent))
import m_computation__gtcnumpy_c8404d4b2c as computation

sys.path = path_backup
del path_backup

from gt4py.definitions import AccessKind, Boundary, CartesianSpace
from gt4py.stencil_object import DomainInfo, FieldInfo, ParameterInfo, StencilObject


class modal2nodal____gtcnumpy_c8404d4b2c(StencilObject):
    """


    The callable interface is the same of the stencil definition function,
    with some extra keyword arguments. Check :class:`gt4py.StencilObject`
    for the full specification.
    """

    _gt_backend_ = "gtc:numpy"

    _gt_source_ = {}

    _gt_domain_info_ = DomainInfo(parallel_axes=("I", "J"), sequential_axis="K", min_sequential_axis_size=0, ndim=3)

    _gt_field_info_ = {
        "vander_inv": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4, 4),
            dtype=dtype("float64"),
        ),
        "in_modal": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
        "out_nodal": FieldInfo(
            access=AccessKind.WRITE,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
    }

    _gt_parameter_info_ = {}

    _gt_constants_ = {}

    _gt_options_ = {
        "name": "modal2nodal",
        "module": "modal_conversion",
        "format_source": True,
        "backend_opts": {},
        "rebuild": True,
        "_impl_opts": {},
    }

    @property
    def backend(self):
        return type(self)._gt_backend_

    @property
    def source(self):
        return type(self)._gt_source_

    @property
    def domain_info(self):
        return type(self)._gt_domain_info_

    @property
    def field_info(self) -> dict:
        return type(self)._gt_field_info_

    @property
    def parameter_info(self) -> dict:
        return type(self)._gt_parameter_info_

    @property
    def constants(self) -> dict:
        return type(self)._gt_constants_

    @property
    def options(self) -> dict:
        return type(self)._gt_options_

    def __call__(self, vander_inv, in_modal, out_nodal, domain=None, origin=None, validate_args=True, exec_info=None):
        if exec_info is not None:
            exec_info["call_start_time"] = time.perf_counter()

        field_args = dict(vander_inv=vander_inv, out_nodal=out_nodal, in_modal=in_modal)
        parameter_args = dict()
        # assert that all required values have been provided

        self._call_run(
            field_args=field_args,
            parameter_args=parameter_args,
            domain=domain,
            origin=origin,
            validate_args=validate_args,
            exec_info=exec_info,
        )

        if exec_info is not None:
            exec_info["call_end_time"] = time.perf_counter()

            if exec_info.setdefault("__aggregate_data", False):
                stencil_info = exec_info.setdefault("modal2nodal____gtcnumpy_c8404d4b2c", {})

                # Update performance counters
                stencil_info["call_start_time"] = exec_info["call_start_time"]
                stencil_info["call_end_time"] = exec_info["call_end_time"]
                stencil_info["call_time"] = stencil_info["call_end_time"] - stencil_info["call_start_time"]
                stencil_info["total_call_time"] = stencil_info.get("total_call_time", 0.0) + stencil_info["call_time"]
                stencil_info["ncalls"] = stencil_info.get("ncalls", 0) + 1
                stencil_info["run_time"] = exec_info["run_end_time"] - exec_info["run_start_time"]
                stencil_info["total_run_time"] = stencil_info.get("total_run_time", 0.0) + stencil_info["run_time"]
                if "run_cpp_start_time" in exec_info:
                    stencil_info["run_cpp_time"] = exec_info["run_cpp_end_time"] - exec_info["run_cpp_start_time"]
                    stencil_info["total_run_cpp_time"] = (
                        stencil_info.get("total_run_cpp_time", 0.0) + stencil_info["run_cpp_time"]
                    )

    def run(
        self,
        _domain_,
        _origin_,
        exec_info,
        *,
        vander_inv,
        out_nodal,
        in_modal,
    ):
        if exec_info is not None:
            exec_info["domain"] = _domain_
            exec_info["origin"] = _origin_
            exec_info["run_start_time"] = time.perf_counter()
        computation.run(
            vander_inv=vander_inv, in_modal=in_modal, out_nodal=out_nodal, _domain_=_domain_, _origin_=_origin_
        )
        if exec_info is not None:
            exec_info["run_end_time"] = time.perf_counter()
