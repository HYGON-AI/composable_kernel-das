# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class CKTileGroupedGemmOperation:
    """CK-side contract for a Torch backend grouped-GEMM consumer.

    This describes the public descriptor-array API rather than pretending to
    be a single-GEMM ``CKTileGemmOperation``.  A consumer must still register a
    grouped lowering/template and call the named C ABI symbols.
    """

    layout_a: str
    layout_b: str
    layout_c: str
    datatype_a: str
    datatype_b: str
    datatype_c: str
    dtype_enum: str
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    pipeline: str
    scheduler: str
    supports_variable_k: bool
    min_k: int
    k_alignment: int
    n_alignment: int
    k_batch: int
    m_padding: str
    public_api_header: str
    run_symbol: str
    workspace_symbol: str
    cmake_target: str

    def layout_repr(self):
        return f"{self.layout_a[0]}{self.layout_b[0]}{self.layout_c[0]}"

    def dtype_repr(self):
        return f"{self.datatype_a}{self.datatype_b}{self.datatype_c}"

    def name(self):
        return "ck_tile_grouped_gemm_" + "_".join(
            [
                self.layout_repr(),
                self.dtype_repr(),
                f"{self.tile_m}{self.tile_n}{self.tile_k}",
                f"{self.warp_m}{self.warp_n}{self.warp_k}",
                f"{self.warp_tile_m}{self.warp_tile_n}{self.warp_tile_k}",
                self.pipeline,
                self.scheduler,
                "VariableK" if self.supports_variable_k else "FixedK",
            ]
        )

    def api_layout_chars(self):
        return self.layout_a[0], self.layout_b[0]

    def supports_problem(self, group_ks: Iterable[int], k_batch: int = 1):
        """Conservative prefilter for the validated FP16/BF16 CompV4 path."""

        ks = tuple(group_ks)
        return (
            bool(ks)
            and k_batch == self.k_batch
            and all(k >= self.min_k and k % self.k_alignment == 0 for k in ks)
        )

    def dict_items(self):
        return asdict(self).items()
