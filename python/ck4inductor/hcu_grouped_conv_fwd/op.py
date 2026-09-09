# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from dataclasses import asdict, dataclass
from typing import Tuple
import hashlib


@dataclass(frozen=True)
class HCUGroupedConvFwdOp:
    """HCU grouped conv fwd instance metadata.

    HCU grouped conv instances are implemented by several MMAC and ck_tile
    wrappers, so this records discovery metadata instead of forcing them into
    the standard AMD XDL grouped-conv shape.
    """

    arch: str
    family: str
    template_name: str
    source_path: str
    a_layout: str
    b_layout: str
    e_layout: str
    a_element_dtype: str
    b_element_dtype: str
    e_element_dtype: str
    conv_forward_specialization: str
    raw_args: Tuple[str, ...]

    def name(self):
        return f"ck_hcu_grouped_conv_fwd_{self.key_name()}"

    def key_name(self):
        prefix = "_".join(
            _format_key_value(value)
            for value in (
                self.arch,
                self.family,
                self.a_layout,
                self.b_layout,
                self.e_layout,
                self.a_element_dtype,
                self.b_element_dtype,
                self.e_element_dtype,
                self.conv_forward_specialization,
            )
        )
        digest = hashlib.sha1("|".join(self.raw_args).encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    def dict_items(self):
        return asdict(self).items()


def _format_key_value(value):
    return str(value).replace("::", "").replace("/", "_").lower()

