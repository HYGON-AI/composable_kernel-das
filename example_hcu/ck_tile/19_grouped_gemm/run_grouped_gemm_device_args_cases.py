#!/usr/bin/env python3
"""从 CSV 批量运行通用 grouped GEMM device-args benchmark，并保存复测证据。

用途
====

本脚本逐行读取 ``grouped_gemm_device_args_cases_example.csv``，把每行参数转换成
``tile_example_grouped_gemm_device_args_benchmark`` 的命令行。它只做运行时参数选择，
不会在运行时实例化 C++ template。实际 kernel 必须已经由 CK 编译进 benchmark。

快速使用
========

1. 先构建 benchmark（构建目录名称可以自定，gfx936/gfx938 均可）：

   ``cmake --build build_device_args_example_bw --target tile_example_grouped_gemm_device_args_benchmark -j 16``

2. 先用 ``--dry-run`` 检查筛选结果和最终命令，不启动 GPU：

   ``python3 run_grouped_gemm_device_args_cases.py --binary=build_device_args_example_bw/bin/tile_example_grouped_gemm_device_args_benchmark --cases=example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_cases_example.csv --output-dir=hygon_tmp/grouped_gemm/dry_run --case-ids=uniform_fp16_nt_g4 --blas-root=/wksp/hipblaslt-install --dry-run``

3. 运行一组或多组正式复测：

   ``python3 run_grouped_gemm_device_args_cases.py --binary=build_device_args_example_bw/bin/tile_example_grouped_gemm_device_args_benchmark --cases=example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_cases_example.csv --output-dir=hygon_tmp/grouped_gemm/results --dtypes=bf16 --layouts=NT,NN,TN --group-counts=1,3,4,5,8 --blas-root=/wksp/hipblaslt-install``

也可以用 ``--case-ids=id1,id2`` 精确选择用例。``--dtypes``、``--layouts``、
``--group-counts`` 和 ``--case-ids`` 同时给出时取交集。G16 占用通常更高，建议与
G1/3/4/5/8 分开执行。CSV 中正式性能行已经设置 10 次 warmup、30 个 GPU event
sample、3 次 correctness；脚本默认以 CK/BLAS CV 不超过 10% 作为通过条件。

BLAS 路径
=========

推荐传 ``--blas-root=/wksp/hipblaslt-install``。benchmark 会同时找到
``lib/libhipblaslt.so`` 及同目录依赖（例如 ``liborigami.so.1``）。只有在系统动态
库搜索路径已经包含所有依赖时，才直接传 ``--blas-lib=/path/libhipblaslt.so``；两个
选项互斥。CSV 每行的 ``backend``、``blas_streams`` 可覆盖脚本默认值。

CSV 字段
========

必需字段为 ``case_id,dtype,layout,m,n,k,group_lengths,variable_capacity``。
``groups`` 用于等长组；``group_lengths`` 用分号分隔，例如 ``65;0;127;129``，此时
``variable_capacity`` 指输入/输出分组维度的总可用容量。``allow_empty_groups=1`` 允许
长度为 0 的组。其余可选字段包括 ``backend,blas_streams,warmup,samples,``
``correctness_repeats``。layout 同时传给 CK 与 BLAS：NT/NN 的动态轴是 M，TN 的动态轴
是 K；脚本不把 layout 解释成某个 consumer 的产品阶段。

工作流程与产物
==============

1. 校验 benchmark、CSV 和 BLAS 参数；读取 CSV 并按筛选条件选行。
2. 计算 benchmark、CSV、hipBLASLt/origami 的路径和 SHA256，写入 metadata。
3. 对每行构造 benchmark 命令；``--dry-run`` 只打印命令。
4. 正式运行时捕获 stdout/stderr，从 SELECT/PERF/RATIO/SUMMARY 结构化行提取
   InstanceId、耗时、TFLOPS、CV、CK/BLAS ratio 和 correctness 结论。
5. 检查子进程退出码、SUMMARY、期望 backend 是否齐全以及 CV 门限。
6. 输出 ``runner.log``（完整命令/日志）、``results.jsonl``（逐用例原始结构化证据）
   和 ``results.csv``（便于聚合比较）；任一用例失败时脚本最终返回非零。

完整 CSV 有 615 行：3 个通用教学/冒烟用例，加上从 Primus PRD 参数矩阵转换出的
612 个通用参数行。CK CSV 不携带 P0/Stage3、FC1/FC2、layer 等 consumer 语义。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = (
    "case_id",
    "dtype",
    "layout",
    "m",
    "n",
    "k",
    "group_lengths",
    "variable_capacity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_evidence(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "resolved_path": None, "sha256": None}
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": sha256(resolved) if resolved.is_file() else None,
    }


def parse_key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def positive(row: dict[str, str], name: str, default: int) -> int:
    value = int(row.get(name) or default)
    if value <= 0:
        raise ValueError(f"{row['case_id']}: {name} must be positive")
    return value


def build_command(args: argparse.Namespace, row: dict[str, str]) -> list[str]:
    # CSV 内使用分号避免和列分隔符冲突；传给 benchmark 时恢复为逗号列表。
    lengths = [item for item in row["group_lengths"].split(";") if item != ""]
    command = [
        str(args.binary),
        f"--m={positive(row, 'm', 0)}",
        f"--n={positive(row, 'n', 0)}",
        f"--k={positive(row, 'k', 0)}",
        f"--dtype={row['dtype']}",
        f"--layout={row['layout']}",
        f"--backend={row.get('backend') or args.backend}",
        f"--warmup={positive(row, 'warmup', args.warmup)}",
        f"--samples={positive(row, 'samples', args.samples)}",
        f"--correctness-repeats={positive(row, 'correctness_repeats', args.correctness_repeats)}",
        f"--include-args-build={args.include_args_build}",
        f"--validate={args.validate}",
        f"--num-cu={args.num_cu}",
        f"--blas-streams={positive(row, 'blas_streams', args.blas_streams)}",
    ]
    if lengths:
        command.extend(
            (
                f"--group-lengths={','.join(lengths)}",
                f"--variable-capacity={positive(row, 'variable_capacity', 0)}",
                f"--allow-empty-groups={int(row.get('allow_empty_groups', '0') or '0')}",
            )
        )
    else:
        command.append(f"--groups={positive(row, 'groups', 1)}")
    if args.blas_lib:
        command.append(f"--blas-lib={args.blas_lib}")
    elif args.blas_root:
        command.append(f"--blas-root={args.blas_root}")
    return command


def main() -> int:
    # 第一步：解析并校验输入，避免跑了很久以后才发现路径或筛选条件错误。
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", help="optional comma-separated case IDs")
    parser.add_argument("--dtypes", help="optional comma-separated dtype filter")
    parser.add_argument("--layouts", help="optional comma-separated layout filter")
    parser.add_argument("--group-counts", help="optional comma-separated group-count filter")
    parser.add_argument("--blas-root")
    parser.add_argument("--blas-lib")
    parser.add_argument("--backend", default="both", choices=("ck", "blas", "both"))
    parser.add_argument("--blas-streams", type=int, default=4, choices=(1, 4))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--correctness-repeats", type=int, default=3)
    parser.add_argument("--include-args-build", type=int, default=1, choices=(0, 1))
    parser.add_argument("--validate", type=int, default=1, choices=(0, 1))
    parser.add_argument("--num-cu", type=int, default=0)
    parser.add_argument("--max-cv-pct", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error(f"binary is not executable: {args.binary}")
    if not args.cases.is_file():
        parser.error(f"cases file does not exist: {args.cases}")
    if args.blas_lib and args.blas_root:
        parser.error("--blas-lib and --blas-root are mutually exclusive")

    selected = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    selected_dtypes = {item.strip() for item in args.dtypes.split(",")} if args.dtypes else None
    selected_layouts = {item.strip() for item in args.layouts.split(",")} if args.layouts else None
    selected_groups = {item.strip() for item in args.group_counts.split(",")} if args.group_counts else None
    # 第二步：加载 CSV，并让多个筛选条件按交集生效。
    with args.cases.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            parser.error(f"missing CSV columns: {','.join(missing)}")
        rows = [
            row
            for row in reader
            if (selected is None or row["case_id"] in selected)
            and (selected_dtypes is None or row["dtype"] in selected_dtypes)
            and (selected_layouts is None or row["layout"] in selected_layouts)
            and (selected_groups is None or row.get("groups", "") in selected_groups)
        ]
    if not rows:
        parser.error("case selection is empty")

    # 第三步：先记录二进制、用例表和 BLAS 依赖的 SHA，保证结果可追溯。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "runner.log"
    jsonl_path = args.output_dir / "results.jsonl"
    csv_path = args.output_dir / "results.csv"
    metadata = {
        "record_type": "metadata",
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256(args.binary),
        "cases": str(args.cases.resolve()),
        "cases_sha256": sha256(args.cases),
        "blas_lib": args.blas_lib,
        "blas_root": args.blas_root,
        "case_count": len(rows),
    }
    blas_path = (
        Path(args.blas_lib)
        if args.blas_lib
        else Path(args.blas_root) / "lib" / "libhipblaslt.so"
        if args.blas_root
        else None
    )
    origami_path = blas_path.parent / "liborigami.so.1" if blas_path is not None else None
    metadata["blas_evidence"] = file_evidence(blas_path)
    metadata["origami_evidence"] = file_evidence(origami_path)
    metadata["host"] = os.uname().nodename if hasattr(os, "uname") else None

    result_rows: list[dict[str, object]] = []
    failures = 0
    with log_path.open("w", encoding="utf-8", newline="\n") as log, jsonl_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as jsonl:
        jsonl.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        # 第四至六步：逐用例执行，解析结构化输出，并执行 correctness/backend/CV 门禁。
        for index, row in enumerate(rows, 1):
            command = build_command(args, row)
            command_text = shlex.join(command)
            log.write(f"===== [{index}/{len(rows)}] {row['case_id']} =====\n")
            log.write(f"COMMAND {command_text}\n")
            log.flush()
            if args.dry_run:
                print(command_text)
                continue
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            output = completed.stdout + completed.stderr
            log.write(output)
            log.write(f"EXIT {completed.returncode}\n")
            log.flush()

            selection: dict[str, str] = {}
            ratio = ""
            summary = "MISSING"
            perf_rows: list[dict[str, str]] = []
            for line in output.splitlines():
                if line.startswith("SELECT "):
                    selection = parse_key_values(line)
                elif line.startswith("PERF "):
                    perf_rows.append(parse_key_values(line))
                elif line.startswith("RATIO "):
                    ratio = parse_key_values(line).get("ck_over_blas", "")
                elif line.startswith("SUMMARY "):
                    summary = parse_key_values(line).get("result", "MISSING")

            requested_backend = row.get("backend") or args.backend
            actual_backends = {perf.get("backend", "") for perf in perf_rows}
            has_ck = "ck_device_args" in actual_backends
            has_blas = any(name.startswith("hipblas") for name in actual_backends)
            expected_backends_present = (
                has_ck and has_blas
                if requested_backend == "both"
                else has_ck
                if requested_backend == "ck"
                else has_blas
            )
            case_failed = (
                completed.returncode != 0
                or summary != "PASS"
                or not expected_backends_present
            )
            for perf in perf_rows:
                cv_pct = float(perf.get("cv", "nan")) * 100.0
                if not math.isfinite(cv_pct) or cv_pct > args.max_cv_pct:
                    case_failed = True
                result_rows.append(
                    {
                        **{key: row.get(key, "") for key in reader.fieldnames or ()},
                        "instance_id": selection.get("instance_id", ""),
                        "instance_name": selection.get("instance_name", ""),
                        "backend_result": perf.get("backend", ""),
                        "median_ms": perf.get("median_ms", ""),
                        "mean_ms": perf.get("mean_ms", ""),
                        "min_ms": perf.get("min_ms", ""),
                        "max_ms": perf.get("max_ms", ""),
                        "cv_pct": cv_pct,
                        "tflops": perf.get("tflops", ""),
                        "ck_over_blas": ratio,
                        "exit_code": completed.returncode,
                        "summary": summary,
                    }
                )
            record = {
                "record_type": "case",
                "case": row,
                "command": command,
                "selection": selection,
                "performance": perf_rows,
                "ck_over_blas": ratio,
                "exit_code": completed.returncode,
                "summary": summary,
                "failed": case_failed,
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()
            failures += int(case_failed)
            print(
                f"CASE case_id={row['case_id']} exit={completed.returncode} "
                f"summary={summary} failed={int(case_failed)}",
                flush=True,
            )

    # 第七步：生成便于汇总的 CSV；JSONL 和完整日志已在循环中同步落盘。
    if result_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
            writer.writeheader()
            writer.writerows(result_rows)
    print(
        f"RUN_SUMMARY cases={len(rows)} failures={failures} "
        f"results_csv={csv_path} jsonl={jsonl_path} log={log_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
