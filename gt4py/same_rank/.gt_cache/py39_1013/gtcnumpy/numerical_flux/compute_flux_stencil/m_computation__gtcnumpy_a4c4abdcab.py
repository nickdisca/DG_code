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


def run(*, u_n, u_s, u_e, u_w, f_n, f_s, f_e, f_w, _domain_, _origin_):

    # --- begin domain boundary shortcuts ---
    _di_, _dj_, _dk_ = 0, 0, 0
    _dI_, _dJ_, _dK_ = _domain_
    # --- end domain padding ---

    u_n = Field(u_n, _origin_["u_n"], (True, True, True))
    u_s = Field(u_s, _origin_["u_s"], (True, True, True))
    u_e = Field(u_e, _origin_["u_e"], (True, True, True))
    u_w = Field(u_w, _origin_["u_w"], (True, True, True))
    f_n = Field(f_n, _origin_["f_n"], (True, True, True))
    f_s = Field(f_s, _origin_["f_s"], (True, True, True))
    f_e = Field(f_e, _origin_["f_e"], (True, True, True))
    f_w = Field(f_w, _origin_["f_w"], (True, True, True))

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):

        # --- begin vertical block ---
        k, K = _dk_, _dK_

        # --- begin horizontal block --
        i, I = _di_ - 0, _dI_ + 0
        j, J = _dj_ - 0, _dJ_ + 0

        flux__a76_3_13__a59_12_42_gen_0 = (
            np.float64((np.int64(1) / np.int64(2)))
            * (f_n[i:I, j:J, k:K] + f_s[i:I, j + 1 : J + 1, k:K])
        ) - (
            np.float64((np.int64(1) / np.int64(2)))
            * (u_s[i:I, j + 1 : J + 1, k:K] - u_n[i:I, j:J, k:K])
        )
        flux_n__a59_12_42_gen_0 = flux__a76_3_13__a59_12_42_gen_0
        flux__5e0_4_13__a59_12_42_gen_0 = (
            np.float64(((-(np.int64(1))) / np.int64(2)))
            * (f_s[i:I, j:J, k:K] + f_n[i:I, j - 1 : J - 1, k:K])
        ) - (
            np.float64((np.int64(1) / np.int64(2)))
            * (u_n[i:I, j - 1 : J - 1, k:K] - u_s[i:I, j:J, k:K])
        )
        flux_s__a59_12_42_gen_0 = flux__5e0_4_13__a59_12_42_gen_0
        flux__5e0_5_13__a59_12_42_gen_0 = (
            np.float64(((-(np.int64(1))) / np.int64(2)))
            * (f_e[i:I, j:J, k:K] + f_w[i:I, j - 1 : J - 1, k:K])
        ) - (
            np.float64((np.int64(1) / np.int64(2)))
            * (u_w[i:I, j - 1 : J - 1, k:K] - u_e[i:I, j:J, k:K])
        )
        flux_e__a59_12_42_gen_0 = flux__5e0_5_13__a59_12_42_gen_0
        flux__5e0_6_13__a59_12_42_gen_0 = (
            np.float64(((-(np.int64(1))) / np.int64(2)))
            * (f_w[i:I, j:J, k:K] + f_e[i:I, j - 1 : J - 1, k:K])
        ) - (
            np.float64((np.int64(1) / np.int64(2)))
            * (u_e[i:I, j - 1 : J - 1, k:K] - u_w[i:I, j:J, k:K])
        )
        flux_w__a59_12_42_gen_0 = flux__5e0_6_13__a59_12_42_gen_0
        flux_n_gen_0 = flux_n__a59_12_42_gen_0
        flux_s_gen_0 = flux_s__a59_12_42_gen_0
        flux_e_gen_0 = flux_e__a59_12_42_gen_0
        flux_w_gen_0 = flux_w__a59_12_42_gen_0
        # --- end horizontal block --

        # --- end vertical block ---
