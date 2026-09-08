# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .._template_parser import TemplateInstance, find_template_instances
from ..util import library_path
from .op import HCUGroupedConvFwdOp

log = logging.getLogger(__name__)

_DIRECT_TEMPLATE_NAMES = (
    "DeviceGroupedConvFwd_mmac_v2_cshuffle_mixed",
    "DeviceGroupedConvFwd_mmac_v2_cshuffle",
    "DeviceGroupedConvFwd_mmac_nchw_v2_cshuffle",
)
_ALIAS_TEMPLATE_NAMES = (
    "DeviceOpF16",
    "DeviceOpBF16",
    "DeviceOpF32",
    "Kernel",
)
_TEMPLATE_NAMES = _DIRECT_TEMPLATE_NAMES + _ALIAS_TEMPLATE_NAMES

_LAYOUT_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_[a-z0-9]+$")
_DTYPE_MAP = {
    "bf16": "BF16",
    "f16": "F16",
    "f32": "F32",
}


def _ck_library_dir() -> Optional[Path]:
    hcu_instances_path = (
        Path(library_path()) / "src" / "tensor_operation_instance_hcu" / "gpu"
    )
    if not hcu_instances_path.exists():
        log.error("CK HCU library path %s does not exist", hcu_instances_path)
        return None
    return hcu_instances_path


def _iter_hcu_grouped_conv_instances(root: Path) -> Iterable[TemplateInstance]:
    for arch_dir in sorted(root.glob("grouped_conv2d_fwd_gfx*")):
        if not arch_dir.is_dir():
            continue
        for template_name in _TEMPLATE_NAMES:
            yield from find_template_instances(arch_dir, template_name, suffixes=(".cpp",))


def _op_from_instance(instance: TemplateInstance, root: Path) -> HCUGroupedConvFwdOp:
    rel_path = instance.path.relative_to(root).as_posix()
    arch = _infer_arch(instance.path)
    a_layout, b_layout, e_layout = _infer_layouts(instance.path)
    a_dtype, b_dtype, e_dtype = _infer_dtypes(instance.path, instance.args)
    return HCUGroupedConvFwdOp(
        arch=arch,
        family=_infer_family(instance),
        template_name=instance.template_name,
        source_path=rel_path,
        a_layout=a_layout,
        b_layout=b_layout,
        e_layout=e_layout,
        a_element_dtype=a_dtype,
        b_element_dtype=b_dtype,
        e_element_dtype=e_dtype,
        conv_forward_specialization=_infer_conv_spec(instance.path, instance.args),
        raw_args=instance.args,
    )


def _infer_arch(path: Path) -> str:
    for part in path.parts:
        if part.startswith("grouped_conv2d_fwd_gfx"):
            return part.rsplit("_", maxsplit=1)[-1]
    return "unknown"


def _infer_layouts(path: Path) -> Tuple[str, str, str]:
    for part in reversed(path.parts):
        if _LAYOUT_RE.match(part):
            layouts = tuple(token.upper() for token in part.split("_"))
            if len(layouts) == 3:
                return layouts
    return ("Unknown", "Unknown", "Unknown")


def _infer_dtypes(path: Path, args: Tuple[str, ...]) -> Tuple[str, str, str]:
    dtype_tokens = [
        _DTYPE_MAP[token]
        for token in re.findall(r"(?:^|_)(bf16|f16|f32)(?=_|$)", path.stem)
        if token in _DTYPE_MAP
    ]
    if not dtype_tokens:
        dtype_tokens = [_dtype_from_arg(arg) for arg in args]
        dtype_tokens = [dtype for dtype in dtype_tokens if dtype]
    if not dtype_tokens:
        return ("Unknown", "Unknown", "Unknown")
    if len(dtype_tokens) == 1:
        return (dtype_tokens[0], dtype_tokens[0], dtype_tokens[0])
    return (
        dtype_tokens[0],
        dtype_tokens[1],
        dtype_tokens[2] if len(dtype_tokens) > 2 else dtype_tokens[0],
    )


def _dtype_from_arg(arg: str) -> Optional[str]:
    normalized = arg.strip()
    if normalized in {"ck::half_t", "F16"}:
        return "F16"
    if normalized in {"ck::bhalf_t", "BF16"}:
        return "BF16"
    if normalized in {"float", "F32"}:
        return "F32"
    return None


def _infer_conv_spec(path: Path, args: Tuple[str, ...]) -> str:
    lower_name = path.stem.lower()
    if "f1x1s1p0" in lower_name:
        return "ConvFwd1x1S1P0"
    if "f1x1p0" in lower_name:
        return "ConvFwd1x1P0"
    if "f3x3" in lower_name:
        return "ConvFwd3x3"
    if "oddc" in lower_name:
        return "ConvFwdOddC"
    for arg in args:
        if arg.startswith("ConvFwd"):
            return arg
    return "ConvFwdDefault"


def _infer_family(instance: TemplateInstance) -> str:
    template_name = instance.template_name
    lower_path = instance.path.as_posix().lower()
    if template_name == "Kernel":
        return "ck_tile_v2r1" if "v2r1" in lower_path else "ck_tile_v1"
    if template_name.startswith("DeviceOp"):
        return "mmac_cshuffle" if "ngchwc" in lower_path or "nhwgc" in lower_path else "mmac"
    if template_name.startswith("DeviceGroupedConvFwd_"):
        return template_name.replace("DeviceGroupedConvFwd_", "")
    return template_name


def parse_instances(instances: Iterable[TemplateInstance], root: Path) -> List[HCUGroupedConvFwdOp]:
    ops = []
    seen = set()
    for instance in instances:
        try:
            op = _op_from_instance(instance, root)
        except ValueError as exc:
            log.debug("%s when parsing %s", exc, instance.path)
            continue
        key = (
            op.source_path,
            op.template_name,
            op.raw_args,
        )
        if key not in seen:
            seen.add(key)
            ops.append(op)
    return ops


@lru_cache(None)
def gen_conv_ops_library() -> List[HCUGroupedConvFwdOp]:
    ck_library_dir = _ck_library_dir()
    if not ck_library_dir:
        return []
    ops = parse_instances(_iter_hcu_grouped_conv_instances(ck_library_dir), ck_library_dir)
    log.debug("ck HCU grouped conv fwd instances from library: %d", len(ops))
    return ops


if __name__ == "__main__":
    for op in gen_conv_ops_library():
        print(op.name())
