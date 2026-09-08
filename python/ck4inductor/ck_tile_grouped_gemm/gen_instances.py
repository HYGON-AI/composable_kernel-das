# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from functools import lru_cache

from .op import CKTileGroupedGemmOperation


@lru_cache(None)
def ops():
    """Return the validated FP16/BF16 NT/NN/TN grouped-GEMM API families."""

    instances = []
    for layout_a, layout_b, layout_c in [
        ("Row", "Col", "Row"),
        ("Row", "Row", "Row"),
        ("Col", "Row", "Row"),
    ]:
        for datatype, dtype_enum in [
            ("FP16", "CK_TILE_HCU_GROUPED_GEMM_FP16"),
            ("BF16", "CK_TILE_HCU_GROUPED_GEMM_BF16"),
        ]:
            instances.append(
                CKTileGroupedGemmOperation(
                    layout_a=layout_a,
                    layout_b=layout_b,
                    layout_c=layout_c,
                    datatype_a=datatype,
                    datatype_b=datatype,
                    datatype_c=datatype,
                    dtype_enum=dtype_enum,
                    tile_m=64,
                    tile_n=128,
                    tile_k=64,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=64,
                    warp_tile_k=32,
                    pipeline="CompV4",
                    scheduler="Intrawave",
                    supports_variable_k=True,
                    min_k=128,
                    k_alignment=1,
                    n_alignment=1,
                    k_batch=1,
                    m_padding="auto",
                    public_api_header="example_hcu/ck_tile/19_grouped_gemm/grouped_gemm.hpp",
                    run_symbol="ck_tile_hcu_grouped_gemm_run",
                    workspace_symbol="ck_tile_hcu_grouped_gemm_workspace_size",
                    cmake_target="ck_tile_hcu_grouped_gemm",
                )
            )
    return instances


if __name__ == "__main__":
    for op in ops():
        print(op.name())
