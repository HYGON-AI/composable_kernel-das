# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Run fixed-length FWD and receipt-3 BWD numerical regressions on one device."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess


def cases(full):
    dtypes = ('fp16', 'bf16')
    masks = ('0', 't', 'b')
    if full:
        for dtype, dim, mask in itertools.product(dtypes, range(1, 129), masks):
            yield 'bwd', dtype, 5, 7, dim, mask, 1, 2, 2
    lengths = [(63, 63), (64, 64), (65, 65), (127, 127), (128, 128), (129, 129),
               (63, 64), (64, 63), (65, 64), (64, 65), (127, 128), (128, 127),
               (129, 128), (128, 129), (63, 129), (129, 63), (65, 127), (127, 65)]
    dims = [32, 56, 57, 64, 72, 120, 121, 128] if full else [64, 120, 128]
    if not full:
        lengths = [(127, 127), (127, 65), (65, 127)]
    for dtype, dim, (sq, sk), mask in itertools.product(dtypes, dims, lengths, masks):
        yield 'bwd', dtype, sq, sk, dim, mask, 1, 2, 2
    for dtype, dim, mask in itertools.product(dtypes, [64, 120, 128], masks):
        yield 'bwd', dtype, 65, 127, dim, mask, 2, 4, 2
    fwd_dims = range(8, 129, 8) if full else [64, 120, 128]
    fwd_lengths = [(1, 1), (5, 7), (63, 65), (65, 63), (127, 65),
                   (65, 127), (127, 127), (128, 128), (129, 129)] if full else lengths
    for dtype, dim, (sq, sk), mask in itertools.product(dtypes, fwd_dims, fwd_lengths, masks):
        yield 'fwd', dtype, sq, sk, dim, mask, 1, 2, 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fwd', type=Path, required=True)
    parser.add_argument('--bwd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (op, dtype, sq, sk, dim, mask, batch, hq, hk) in enumerate(cases(args.full)):
        name = f'{op}-{dtype}-q{sq}-k{sk}-d{dim}-m{mask}-b{batch}-h{hq}-hk{hk}'
        command = [str(getattr(args, op).resolve()), '-v=1', '-p_drop=0', '-seed=142',
                   '-warmup=0', '-repeat=1', '-kname=1', f'-prec={dtype}',
                   f'-s={sq}', f'-s_k={sk}', f'-d={dim}', f'-mask={mask}',
                   f'-b={batch}', f'-h={hq}', f'-h_k={hk}']
        if op == 'fwd':
            command.append('-lse=1')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False)
        (args.output / (name + '.log')).write_text(' '.join(command) + '\n' + result.stdout)
        valid = result.returncode == 0 and 'valid:y' in result.stdout
        rows.append({'case': name, 'exit': result.returncode, 'valid': valid})
        if not valid or index % 100 == 0:
            print(json.dumps(rows[-1]), flush=True)
    (args.output / 'results.json').write_text(json.dumps(rows, indent=2))
    print(f'TOTAL {len(rows)} PASS {sum(r["valid"] for r in rows)}', flush=True)
    return 0 if all(r['valid'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
