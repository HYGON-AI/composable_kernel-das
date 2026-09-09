# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class CKGroupedConvFwdOp:
    """HCU DeviceGroupedConvFwdMultipleD_Xdl_CShuffle template parameters."""

    n_dim_spatial: int
    a_layout: str
    b_layout: str
    ds_layout: Tuple[str, ...]
    e_layout: str
    a_element_dtype: str
    b_element_dtype: str
    acc_dtype: str
    c_shuffle_dtype: str
    ds_element_dtype: Tuple[str, ...]
    e_element_dtype: str
    a_elementwise_op: str
    b_elementwise_op: str
    cde_elementwise_op: str
    conv_forward_specialization: str
    gemm_specialization: str
    num_gemm_k_prefetch_stage: int
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    a_k1: int
    b_k1: int
    m_per_xdl: int
    n_per_xdl: int
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    a_block_transfer_thread_cluster_lengths_ak0_m_ak1: Tuple[int, ...]
    a_block_transfer_thread_cluster_arrange_order: Tuple[int, ...]
    a_block_transfer_src_access_order: Tuple[int, ...]
    a_block_transfer_src_vector_dim: int
    a_block_transfer_src_scalar_per_vector: int
    a_block_transfer_dst_scalar_per_vector_ak1: int
    a_block_lds_extra_m: bool
    b_block_transfer_thread_cluster_lengths_bk0_n_bk1: Tuple[int, ...]
    b_block_transfer_thread_cluster_arrange_order: Tuple[int, ...]
    b_block_transfer_src_access_order: Tuple[int, ...]
    b_block_transfer_src_vector_dim: int
    b_block_transfer_src_scalar_per_vector: int
    b_block_transfer_dst_scalar_per_vector_bk1: int
    b_block_lds_extra_n: bool
    c_shuffle_m_xdl_per_wave_per_shuffle: int
    c_shuffle_n_xdl_per_wave_per_shuffle: int
    cde_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block: Tuple[int, ...]
    cde_block_transfer_scalar_per_vector_n_per_block: int
    loop_scheduler: str = "LoopScheduler::Default"

    def name(self):
        return f"ck_device_grouped_conv_fwd_multiple_d_xdl_cshuffle_{self.key_name()}"

    def key_name(self):
        return "_".join(
            "K" + field_name.replace("_", "").lower() + "V" + _format_key_value(field_value)
            for field_name, field_value in self.dict_items()
        )

    def dict_items(self):
        return asdict(self).items()


def _format_key_value(value):
    if isinstance(value, tuple):
        return "x".join(map(str, value))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("::", "")
