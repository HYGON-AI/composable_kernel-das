# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Run the real FMHA runner and require numerical success plus the intended API."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', default='fp16,bf16')
    parser.add_argument('--hdim', default='64,128')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for dtype in args.dtype.split(','):
        for dim in map(int, args.hdim.split(',')):
            cases = [
                ('ordinary', 'fmha_fwd_d',
                 dict(s=16, s_k=128)),
                ('split_even', 'fmha_fwd_splitkv_d',
                 dict(s=16, s_k=256, num_splits=2, lse=1)),
                ('split_uneven', 'fmha_fwd_splitkv_d',
                 dict(s=5, s_k=129, num_splits=3, lse=1)),
                ('split_decode', 'fmha_fwd_splitkv_d',
                 dict(s=1, s_k=257, num_splits=3, lse=1)),
                ('split_causal', 'fmha_fwd_splitkv_d',
                 dict(s=17, s_k=129, num_splits=3, mask='t', lse=1)),
                ('split_window', 'fmha_fwd_splitkv_d',
                 dict(s=33, s_k=257, num_splits=5, mask='b:29,0', lse=1)),
                ('split_empty', 'fmha_fwd_splitkv_d',
                 dict(s=17, s_k=3, num_splits=8, mask='b', lse=1)),
                ('split_tail_dim', 'fmha_fwd_splitkv_d',
                 dict(s=5, s_k=129, num_splits=3, d=dim - 8, d_v=dim - 16, lse=1)),
                ('split_bias', 'fmha_fwd_splitkv_d',
                 dict(s=17, s_k=129, num_splits=3, bias='e:2', lse=1)),
                ('split_softcap', 'fmha_fwd_splitkv_d',
                 dict(s=17, s_k=129, num_splits=3, logits_soft_cap=2, mask='b', lse=1)),
                ('split_softcap_bias', 'fmha_fwd_splitkv_d',
                 dict(s=17, s_k=129, num_splits=3, logits_soft_cap=2, bias='e:2', mask='b', lse=1)),
                ('split_paged', 'fmha_fwd_splitkv_d',
                 dict(s=5, s_k=257, num_splits=3, page_block_size=128, lse=1)),
                ('split_cache_map', 'fmha_fwd_splitkv_d',
                 dict(s=5, s_k=129, num_splits=3, cache_batch_idx=1, lse=1)),
                ('paged', 'fmha_fwd_pagedkv_d',
                 dict(s=33, s_k=257, page_block_size=128)),
                ('paged_causal', 'fmha_fwd_pagedkv_d',
                 dict(s=17, s_k=129, page_block_size=128, mask='b')),
                ('paged_window', 'fmha_fwd_pagedkv_d',
                 dict(s=33, s_k=257, page_block_size=128, mask='b:29,0')),
                ('paged_group', 'fmha_fwd_pagedkv_d',
                 dict(s='5,17', s_k='129,257', mode=1, page_block_size=128, mask='b')),
                ('paged_bias', 'fmha_fwd_pagedkv_d',
                 dict(s=17, s_k=129, page_block_size=128, bias='e:2')),
                ('paged_softcap', 'fmha_fwd_pagedkv_d',
                 dict(s=17, s_k=129, page_block_size=128, logits_soft_cap=2, mask='b')),
                ('paged_softcap_bias', 'fmha_fwd_pagedkv_d',
                 dict(s=17, s_k=129, page_block_size=128, logits_soft_cap=2, bias='e:2', mask='b')),
                ('append', 'fmha_fwd_appendkv_d',
                 dict(s=5, s_k=141, s_knew=17, num_splits=3)),
                ('append_inter', 'fmha_fwd_appendkv_d',
                 dict(s=5, s_k=141, s_knew=17, num_splits=3, rotary_dim=dim, rotary_interleaved=1)),
                ('append_half', 'fmha_fwd_appendkv_d',
                 dict(s=5, s_k=141, s_knew=17, num_splits=3, rotary_dim=dim, rotary_interleaved=0)),
                ('append_paged', 'fmha_fwd_appendkv_d',
                 dict(s=5, s_k=141, s_knew=17, page_block_size=128)),
                ('append_paged_inter', 'fmha_fwd_appendkv_d',
                 dict(s=5, s_k=141, s_knew=17, page_block_size=128, rotary_dim=dim, rotary_interleaved=1, mask='b')),
                ('append_paged_half', 'fmha_fwd_appendkv_d',
                 dict(s=5, s_k=141, s_knew=17, page_block_size=128, rotary_dim=dim, rotary_interleaved=0, mask='b')),
            ]
            for name, api, params in cases:
                label = f'{dtype}_d{dim}_{name}'
                opts = dict(v=1, b=2, h=4, h_k=2, d=dim, prec=dtype, time=0, kname=1, seed=11939, num_splits=1)
                opts.update(params)
                cmd = [str(args.binary.resolve())] + [f'-{k}={v}' for k, v in opts.items()]
                try:
                    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
                    output = r.stdout
                    code = r.returncode
                except subprocess.TimeoutExpired as e:
                    output = str(e)
                    code = 124
                passed = code == 0 and 'valid:y' in output and (api in output)
                (args.output / f'{label}.log').write_text(output)
                results.append(dict(case=label, passed=passed, exit_code=code, command=cmd, expected_api=api))
                print(f"{('PASS' if passed else 'FAIL')} {label}", flush=True)
                if not passed:
                    print(output[:1600], flush=True)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    failed = sum((not r['passed'] for r in results))
    print(f'FMHA_MATRIX cases={len(results)} failures={failed}', flush=True)
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
