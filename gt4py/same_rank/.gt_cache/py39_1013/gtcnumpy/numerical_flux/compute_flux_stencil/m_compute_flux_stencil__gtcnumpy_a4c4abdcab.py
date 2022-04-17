import pathlib
import time

import numpy as np
from numpy import dtype
import sys
import pathlib
import numpy

path_backup = sys.path.copy()
sys.path.append(str(pathlib.Path(__file__).parent))
import m_computation__gtcnumpy_a4c4abdcab as computation

sys.path = path_backup
del path_backup

from gt4py.definitions import AccessKind, Boundary, CartesianSpace
from gt4py.stencil_object import DomainInfo, FieldInfo, ParameterInfo, StencilObject


class compute_flux_stencil____gtcnumpy_a4c4abdcab(StencilObject):
    """


    The callable interface is the same of the stencil definition function,
    with some extra keyword arguments. Check :class:`gt4py.StencilObject`
    for the full specification.
    """

    _gt_backend_ = "gtc:numpy"

    _gt_source_ = {}

    _gt_domain_info_ = DomainInfo(parallel_axes=("I", "J"), sequential_axis="K", min_sequential_axis_size=0, ndim=3)

    _gt_field_info_ = {
        "u_n": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (1, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "u_s": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 1), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "u_e": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (1, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "u_w": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (1, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "f_n": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (1, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "f_s": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 1), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "f_e": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (1, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
        "f_w": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (1, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(2,),
            dtype=dtype("float64"),
        ),
    }

    _gt_parameter_info_ = {}

    _gt_constants_ = {}

    _gt_options_ = {
        "name": "compute_flux_stencil",
        "module": "numerical_flux",
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

    def __call__(
        self, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w, domain=None, origin=None, validate_args=True, exec_info=None
    ):
        if exec_info is not None:
            exec_info["call_start_time"] = time.perf_counter()

        field_args = dict(f_n=f_n, u_s=u_s, u_w=u_w, f_s=f_s, u_n=u_n, f_w=f_w, u_e=u_e, f_e=f_e)
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
                stencil_info = exec_info.setdefault("compute_flux_stencil____gtcnumpy_a4c4abdcab", {})

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
        f_n,
        u_s,
        u_w,
        f_s,
        u_n,
        f_w,
        u_e,
        f_e,
    ):
        if exec_info is not None:
            exec_info["domain"] = _domain_
            exec_info["origin"] = _origin_
            exec_info["run_start_time"] = time.perf_counter()
        computation.run(
            u_n=u_n, u_s=u_s, u_e=u_e, u_w=u_w, f_n=f_n, f_s=f_s, f_e=f_e, f_w=f_w, _domain_=_domain_, _origin_=_origin_
        )
        if exec_info is not None:
            exec_info["run_end_time"] = time.perf_counter()
