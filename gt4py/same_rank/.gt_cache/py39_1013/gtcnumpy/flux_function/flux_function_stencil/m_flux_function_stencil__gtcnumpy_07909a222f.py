import pathlib
import time

import numpy as np
from numpy import dtype
import sys
import pathlib
import numpy

path_backup = sys.path.copy()
sys.path.append(str(pathlib.Path(__file__).parent))
import m_computation__gtcnumpy_07909a222f as computation

sys.path = path_backup
del path_backup

from gt4py.definitions import AccessKind, Boundary, CartesianSpace
from gt4py.stencil_object import DomainInfo, FieldInfo, ParameterInfo, StencilObject


class flux_function_stencil____gtcnumpy_07909a222f(StencilObject):
    """


    The callable interface is the same of the stencil definition function,
    with some extra keyword arguments. Check :class:`gt4py.StencilObject`
    for the full specification.
    """

    _gt_backend_ = "gtc:numpy"

    _gt_source_ = {}

    _gt_domain_info_ = DomainInfo(parallel_axes=("I", "J"), sequential_axis="K", min_sequential_axis_size=0, ndim=3)

    _gt_field_info_ = {
        "u": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
        "fx": FieldInfo(
            access=AccessKind.WRITE,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
        "fy": FieldInfo(
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
        "name": "flux_function_stencil",
        "module": "flux_function",
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

    def __call__(self, u, fx, fy, domain=None, origin=None, validate_args=True, exec_info=None):
        if exec_info is not None:
            exec_info["call_start_time"] = time.perf_counter()

        field_args = dict(u=u, fy=fy, fx=fx)
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
                stencil_info = exec_info.setdefault("flux_function_stencil____gtcnumpy_07909a222f", {})

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
        u,
        fy,
        fx,
    ):
        if exec_info is not None:
            exec_info["domain"] = _domain_
            exec_info["origin"] = _origin_
            exec_info["run_start_time"] = time.perf_counter()
        computation.run(u=u, fx=fx, fy=fy, _domain_=_domain_, _origin_=_origin_)
        if exec_info is not None:
            exec_info["run_end_time"] = time.perf_counter()
