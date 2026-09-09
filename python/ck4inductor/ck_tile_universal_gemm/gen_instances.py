# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

from functools import lru_cache

from .op import CKTileGemmOperation


@lru_cache(None)
def ops():
    """Generate the HCU ck_tile GEMM candidates with a conservative allowlist."""

    instances = []

    for layout_a, layout_b, layout_c in [
        ("Row", "Row", "Row"),
        ("Row", "Col", "Row"),
    ]:
        for datatype_a, datatype_b, datatype_c in [("FP16",) * 3, ("BF16",) * 3]:
            for m_is_padded in ["true", "false"]:
                for n_is_padded in ["true", "false"]:
                    for k_is_padded in ["true", "false"]:
                        instances.append(
                            CKTileGemmOperation(
                                layout_a=layout_a,
                                layout_b=layout_b,
                                layout_c=layout_c,
                                datatype_a=datatype_a,
                                datatype_b=datatype_b,
                                datatype_c=datatype_c,
                                tile_m=128,
                                tile_n=128,
                                tile_k=32,
                                warp_m=2,
                                warp_n=2,
                                warp_k=1,
                                warp_tile_m=32,
                                warp_tile_n=32,
                                warp_tile_k=8,
                                m_is_padded=m_is_padded,
                                n_is_padded=n_is_padded,
                                k_is_padded=k_is_padded,
                                pipeline="Mem",
                                scheduler="Intrawave",
                                epilogue="Default",
                            )
                        )

                        instances.append(
                            CKTileGemmOperation(
                                layout_a=layout_a,
                                layout_b=layout_b,
                                layout_c=layout_c,
                                datatype_a=datatype_a,
                                datatype_b=datatype_b,
                                datatype_c=datatype_c,
                                tile_m=64,
                                tile_n=128,
                                tile_k=64 if datatype_a in {"FP16", "BF16"} else 32,
                                warp_m=2,
                                warp_n=2,
                                warp_k=1,
                                warp_tile_m=16,
                                warp_tile_n=64,
                                warp_tile_k=32,
                                m_is_padded=m_is_padded,
                                n_is_padded=n_is_padded,
                                k_is_padded=k_is_padded,
                                pipeline="CompV4",
                                scheduler="Intrawave",
                                epilogue="CShuffle",
                            )
                        )

    return instances


if __name__ == "__main__":
    for op in ops():
        print(op.name())
