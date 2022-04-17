import pathlib
import time

import numpy as np
from numpy import dtype
import sys
import pathlib
import numpy

path_backup = sys.path.copy()
sys.path.append(str(pathlib.Path(__file__).parent))
import m_computation__gtcnumpy_62b9f88fdb as computation

sys.path = path_backup
del path_backup

from gt4py.definitions import AccessKind, Boundary, CartesianSpace
from gt4py.stencil_object import DomainInfo, FieldInfo, ParameterInfo, StencilObject


class integrate_flux_stencil____gtcnumpy_62b9f88fdb(StencilObject):
    """


    The callable interface is the same of the stencil definition function,
    with some extra keyword arguments. Check :class:`gt4py.StencilObject`
    for the full specification.
    """

    _gt_backend_ = "gtc:numpy"

    _gt_source_ = {}

    _gt_domain_info_ = DomainInfo(parallel_axes=("I", "J"), sequential_axis="K", min_sequential_axis_size=0, ndim=3)

    _gt_field_info_ = {
        "w": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
        "fx": FieldInfo(
            access=AccessKind.READ_WRITE,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
        "fy": FieldInfo(
            access=AccessKind.READ_WRITE,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
        "phi_grad_x": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4, 4),
            dtype=dtype("float64"),
        ),
        "phi_grad_y": FieldInfo(
            access=AccessKind.READ,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4, 4),
            dtype=dtype("float64"),
        ),
        "rhs": FieldInfo(
            access=AccessKind.WRITE,
            boundary=Boundary(((0, 0), (0, 0), (0, 0))),
            axes=("I", "J", "K"),
            data_dims=(4,),
            dtype=dtype("float64"),
        ),
    }

    _gt_parameter_info_ = {
        "determ": ParameterInfo(access=AccessKind.READ, dtype=dtype("float64")),
        "bd_det_x": ParameterInfo(access=AccessKind.READ, dtype=dtype("float64")),
        "bd_det_y": ParameterInfo(access=AccessKind.READ, dtype=dtype("float64")),
    }

    _gt_constants_ = {}

    _gt_options_ = {
        "name": "integrate_flux_stencil",
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

    def __call__(
        self,
        w,
        fx,
        fy,
        phi_grad_x,
        phi_grad_y,
        rhs,
        determ,
        bd_det_x,
        bd_det_y,
        domain=None,
        origin=None,
        validate_args=True,
        exec_info=None,
    ):
        if exec_info is not None:
            exec_info["call_start_time"] = time.perf_counter()

        field_args = dict(fy=fy, phi_grad_y=phi_grad_y, fx=fx, w=w, phi_grad_x=phi_grad_x, rhs=rhs)
        parameter_args = dict(bd_det_y=bd_det_y, determ=determ, bd_det_x=bd_det_x)
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
                stencil_info = exec_info.setdefault("integrate_flux_stencil____gtcnumpy_62b9f88fdb", {})

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

    def run(self, _domain_, _origin_, exec_info, *, fy, phi_grad_y, fx, w, phi_grad_x, rhs, bd_det_y, determ, bd_det_x):
        if exec_info is not None:
            exec_info["domain"] = _domain_
            exec_info["origin"] = _origin_
            exec_info["run_start_time"] = time.perf_counter()
        computation.run(
            w=w,
            fx=fx,
            fy=fy,
            phi_grad_x=phi_grad_x,
            phi_grad_y=phi_grad_y,
            rhs=rhs,
            determ=determ,
            bd_det_x=bd_det_x,
            bd_det_y=bd_det_y,
            _domain_=_domain_,
            _origin_=_origin_,
        )
        if exec_info is not None:
            exec_info["run_end_time"] = time.perf_counter()
