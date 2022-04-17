import numbers
from typing import Tuple

import numpy as np
import scipy.special


class Field:
    def __init__(self, field, offsets: Tuple[int, ...], dimensions: Tuple[bool, bool, bool]):
        ii = iter(range(3))
        self.idx_to_data = tuple(
            [next(ii) if has_dim else None for has_dim in dimensions]
            + list(range(sum(dimensions), len(field.shape)))
        )

        shape = [field.shape[i] if i is not None else 1 for i in self.idx_to_data]
        self.field_view = np.reshape(field.data, shape).view(np.ndarray)

        self.offsets = offsets

    @classmethod
    def empty(cls, shape, offset):
        return cls(np.empty(shape), offset, (True, True, True))

    def shim_key(self, key):
        new_args = []
        if not isinstance(key, tuple):
            key = (key,)
        for index in self.idx_to_data:
            if index is None:
                new_args.append(slice(None, None))
            else:
                idx = key[index]
                offset = self.offsets[index]
                if isinstance(idx, slice):
                    new_args.append(
                        slice(idx.start + offset, idx.stop + offset, idx.step) if offset else idx
                    )
                else:
                    new_args.append(idx + offset)
        if not isinstance(new_args[2], (numbers.Integral, slice)):
            new_args = self.broadcast_and_clip_variable_k(new_args)
        return tuple(new_args)

    def broadcast_and_clip_variable_k(self, new_args: tuple):
        assert isinstance(new_args[0], slice) and isinstance(new_args[1], slice)
        if np.max(new_args[2]) >= self.field_view.shape[2] or np.min(new_args[2]) < 0:
            new_args[2] = np.clip(new_args[2].copy(), 0, self.field_view.shape[2] - 1)
        new_args[:2] = np.broadcast_arrays(
            np.expand_dims(
                np.arange(new_args[0].start, new_args[0].stop),
                axis=tuple(i for i in range(self.field_view.ndim) if i != 0),
            ),
            np.expand_dims(
                np.arange(new_args[1].start, new_args[1].stop),
                axis=tuple(i for i in range(self.field_view.ndim) if i != 1),
            ),
        )
        return new_args

    def __getitem__(self, key):
        return self.field_view.__getitem__(self.shim_key(key))

    def __setitem__(self, key, value):
        return self.field_view.__setitem__(self.shim_key(key), value)


def run(*, w, fx, fy, phi_grad_x, phi_grad_y, rhs, _domain_, _origin_):

    # --- begin domain boundary shortcuts ---
    _di_, _dj_, _dk_ = 0, 0, 0
    _dI_, _dJ_, _dK_ = _domain_
    # --- end domain padding ---

    w = Field(w, _origin_["w"], (True, True, True))
    fx = Field(fx, _origin_["fx"], (True, True, True))
    fy = Field(fy, _origin_["fy"], (True, True, True))
    phi_grad_x = Field(phi_grad_x, _origin_["phi_grad_x"], (True, True, True))
    phi_grad_y = Field(phi_grad_y, _origin_["phi_grad_y"], (True, True, True))
    rhs = Field(rhs, _origin_["rhs"], (True, True, True))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):

        # --- begin vertical block ---
        k, K = _dk_, _dK_

        # --- begin horizontal block --
        i, I = _di_ - 0, _dI_ + 0
        j, J = _dj_ - 0, _dJ_ + 0

        fx[i:I, j:J, k:K, 0] = fx[i:I, j:J, k:K, 0] * w[i:I, j:J, k:K, 0]
        fx[i:I, j:J, k:K, 1] = fx[i:I, j:J, k:K, 1] * w[i:I, j:J, k:K, 1]
        fx[i:I, j:J, k:K, 2] = fx[i:I, j:J, k:K, 2] * w[i:I, j:J, k:K, 2]
        fx[i:I, j:J, k:K, 3] = fx[i:I, j:J, k:K, 3] * w[i:I, j:J, k:K, 3]
        a_0__00b_15_29_gen_0 = (
            (
                (phi_grad_x[i:I, j:J, k:K, 0, 0] * fx[i:I, j:J, k:K, 0])
                + (phi_grad_x[i:I, j:J, k:K, 1, 0] * fx[i:I, j:J, k:K, 1])
            )
            + (phi_grad_x[i:I, j:J, k:K, 2, 0] * fx[i:I, j:J, k:K, 2])
        ) + (phi_grad_x[i:I, j:J, k:K, 3, 0] * fx[i:I, j:J, k:K, 3])
        a_1__00b_15_29_gen_0 = (
            (
                (phi_grad_x[i:I, j:J, k:K, 0, 1] * fx[i:I, j:J, k:K, 0])
                + (phi_grad_x[i:I, j:J, k:K, 1, 1] * fx[i:I, j:J, k:K, 1])
            )
            + (phi_grad_x[i:I, j:J, k:K, 2, 1] * fx[i:I, j:J, k:K, 2])
        ) + (phi_grad_x[i:I, j:J, k:K, 3, 1] * fx[i:I, j:J, k:K, 3])
        a_2__00b_15_29_gen_0 = (
            (
                (phi_grad_x[i:I, j:J, k:K, 0, 2] * fx[i:I, j:J, k:K, 0])
                + (phi_grad_x[i:I, j:J, k:K, 1, 2] * fx[i:I, j:J, k:K, 1])
            )
            + (phi_grad_x[i:I, j:J, k:K, 2, 2] * fx[i:I, j:J, k:K, 2])
        ) + (phi_grad_x[i:I, j:J, k:K, 3, 2] * fx[i:I, j:J, k:K, 3])
        a_3__00b_15_29_gen_0 = (
            (
                (phi_grad_x[i:I, j:J, k:K, 0, 3] * fx[i:I, j:J, k:K, 0])
                + (phi_grad_x[i:I, j:J, k:K, 1, 3] * fx[i:I, j:J, k:K, 1])
            )
            + (phi_grad_x[i:I, j:J, k:K, 2, 3] * fx[i:I, j:J, k:K, 2])
        ) + (phi_grad_x[i:I, j:J, k:K, 3, 3] * fx[i:I, j:J, k:K, 3])
        x_0_gen_0 = a_0__00b_15_29_gen_0
        x_1_gen_0 = a_1__00b_15_29_gen_0
        x_2_gen_0 = a_2__00b_15_29_gen_0
        x_3_gen_0 = a_3__00b_15_29_gen_0
        fy[i:I, j:J, k:K, 0] = fy[i:I, j:J, k:K, 0] * w[i:I, j:J, k:K, 0]
        fy[i:I, j:J, k:K, 1] = fy[i:I, j:J, k:K, 1] * w[i:I, j:J, k:K, 1]
        fy[i:I, j:J, k:K, 2] = fy[i:I, j:J, k:K, 2] * w[i:I, j:J, k:K, 2]
        fy[i:I, j:J, k:K, 3] = fy[i:I, j:J, k:K, 3] * w[i:I, j:J, k:K, 3]
        a_0__00b_22_29_gen_0 = (
            (
                (phi_grad_y[i:I, j:J, k:K, 0, 0] * fy[i:I, j:J, k:K, 0])
                + (phi_grad_y[i:I, j:J, k:K, 1, 0] * fy[i:I, j:J, k:K, 1])
            )
            + (phi_grad_y[i:I, j:J, k:K, 2, 0] * fy[i:I, j:J, k:K, 2])
        ) + (phi_grad_y[i:I, j:J, k:K, 3, 0] * fy[i:I, j:J, k:K, 3])
        a_1__00b_22_29_gen_0 = (
            (
                (phi_grad_y[i:I, j:J, k:K, 0, 1] * fy[i:I, j:J, k:K, 0])
                + (phi_grad_y[i:I, j:J, k:K, 1, 1] * fy[i:I, j:J, k:K, 1])
            )
            + (phi_grad_y[i:I, j:J, k:K, 2, 1] * fy[i:I, j:J, k:K, 2])
        ) + (phi_grad_y[i:I, j:J, k:K, 3, 1] * fy[i:I, j:J, k:K, 3])
        a_2__00b_22_29_gen_0 = (
            (
                (phi_grad_y[i:I, j:J, k:K, 0, 2] * fy[i:I, j:J, k:K, 0])
                + (phi_grad_y[i:I, j:J, k:K, 1, 2] * fy[i:I, j:J, k:K, 1])
            )
            + (phi_grad_y[i:I, j:J, k:K, 2, 2] * fy[i:I, j:J, k:K, 2])
        ) + (phi_grad_y[i:I, j:J, k:K, 3, 2] * fy[i:I, j:J, k:K, 3])
        a_3__00b_22_29_gen_0 = (
            (
                (phi_grad_y[i:I, j:J, k:K, 0, 3] * fy[i:I, j:J, k:K, 0])
                + (phi_grad_y[i:I, j:J, k:K, 1, 3] * fy[i:I, j:J, k:K, 1])
            )
            + (phi_grad_y[i:I, j:J, k:K, 2, 3] * fy[i:I, j:J, k:K, 2])
        ) + (phi_grad_y[i:I, j:J, k:K, 3, 3] * fy[i:I, j:J, k:K, 3])
        y_0_gen_0 = a_0__00b_22_29_gen_0
        y_1_gen_0 = a_1__00b_22_29_gen_0
        y_2_gen_0 = a_2__00b_22_29_gen_0
        y_3_gen_0 = a_3__00b_22_29_gen_0
        rhs[i:I, j:J, k:K, 0] = x_0_gen_0 + y_0_gen_0
        rhs[i:I, j:J, k:K, 1] = x_1_gen_0 + y_1_gen_0
        rhs[i:I, j:J, k:K, 2] = x_2_gen_0 + y_2_gen_0
        rhs[i:I, j:J, k:K, 3] = x_3_gen_0 + y_3_gen_0
        # --- end horizontal block --

        # --- end vertical block ---
