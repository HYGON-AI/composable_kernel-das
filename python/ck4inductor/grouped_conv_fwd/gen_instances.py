# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from .._template_parser import TemplateInstance, find_template_instances
from ..util import library_path
from .op import CKGroupedConvFwdOp

log = logging.getLogger(__name__)

_TEMPLATE_NAME = "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle"
_EXPECTED_ARGS_WITH_DEFAULT_LOOP = 45
_EXPECTED_ARGS_WITH_EXPLICIT_LOOP = 46

_LAYOUT_SETS = (
    ("NHWGC", "GKYXC", "NHWGK"),
    ("GNHWC", "GKYXC", "GNHWK"),
)
_CONV_SPECS = (
    "ConvFwdDefault",
    "ConvFwd1x1P0",
    "ConvFwd1x1S1P0",
    "ConvFwdOddC",
)


def _ck_library_dir():
    conv_instances_path = (
        Path(library_path()) / "src" / "tensor_operation_instance" / "gpu" / "grouped_conv2d_fwd"
    )
    if not conv_instances_path.exists():
        log.error("CK library path %s does not exist", conv_instances_path)
        return None
    return conv_instances_path


def _iter_grouped_conv_xdl_instances(root: Path) -> Iterable[TemplateInstance]:
    for instance in find_template_instances(root, _TEMPLATE_NAME, suffixes=(".hpp", ".cpp")):
        if instance.path.name.startswith("device_grouped_conv2d_fwd_xdl"):
            yield instance


def _convert_token(token: str):
    token = token.strip()
    if token.startswith("S<") and token.endswith(">"):
        inner = token[2:-1].strip()
        if not inner:
            return tuple()
        return tuple(int(part.strip()) for part in inner.split(","))
    if token in {"DsLayout", "DsDatatype", "Empty_Tuple", "Tuple<>"}:
        return tuple()
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    try:
        return int(token)
    except ValueError:
        return token


def _bool_token(token: str):
    value = _convert_token(token)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return value


def _op_from_instance(instance: TemplateInstance) -> CKGroupedConvFwdOp:
    args = list(instance.args)
    if len(args) == _EXPECTED_ARGS_WITH_DEFAULT_LOOP:
        args.append("LoopScheduler::Default")
    if len(args) != _EXPECTED_ARGS_WITH_EXPLICIT_LOOP:
        raise ValueError(f"{instance.path}: expected 45 or 46 args for {_TEMPLATE_NAME}, got {len(args)}")

    values = [_convert_token(arg) for arg in args]
    values[3] = tuple()
    values[9] = tuple()
    values[33] = _bool_token(args[33])
    values[40] = _bool_token(args[40])
    return CKGroupedConvFwdOp(*values)


def parse_instances(instances: Iterable[TemplateInstance]) -> List[CKGroupedConvFwdOp]:
    ops = []
    for instance in instances:
        try:
            ops.append(_op_from_instance(instance))
        except (TypeError, ValueError) as exc:
            log.debug("%s when parsing %s", exc, instance.path)
    return ops


def _substitute_placeholders(instance: CKGroupedConvFwdOp) -> Iterable[CKGroupedConvFwdOp]:
    conv_specs = _CONV_SPECS if instance.conv_forward_specialization == "ConvSpec" else (
        instance.conv_forward_specialization,
    )
    cde_ops = ("PassThrough",) if instance.cde_elementwise_op == "CDEElementOp" else (
        instance.cde_elementwise_op,
    )
    for a_layout, b_layout, e_layout in _LAYOUT_SETS:
        for conv_spec in conv_specs:
            for cde_op in cde_ops:
                yield replace(
                    instance,
                    n_dim_spatial=2,
                    a_layout=a_layout,
                    b_layout=b_layout,
                    ds_layout=tuple(),
                    e_layout=e_layout,
                    ds_element_dtype=tuple(),
                    cde_elementwise_op=cde_op,
                    conv_forward_specialization=conv_spec,
                    gemm_specialization="GemmMNKPadding",
                )


@lru_cache(None)
def gen_conv_ops_library() -> List[CKGroupedConvFwdOp]:
    ck_library_dir = _ck_library_dir()
    if not ck_library_dir:
        return []
    raw_instances = parse_instances(_iter_grouped_conv_xdl_instances(ck_library_dir))
    op_instances = [op for instance in raw_instances for op in _substitute_placeholders(instance)]
    log.debug("ck grouped conv fwd instances from library: %d", len(op_instances))
    return op_instances


if __name__ == "__main__":
    for op in gen_conv_ops_library():
        print(op.name())
