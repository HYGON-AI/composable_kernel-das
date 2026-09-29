# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""HCU architecture and launch-contract regression for extended FMHA codegen.

Run with python -B test_hcu_codegen.py. This tests generated source contracts;
it does not claim numerical support for the extended attention pipelines.
"""
import unittest

from codegen.arch import get_factories_for_targets, get_hcu_arch_tag
from codegen.ops import (
    fmha_bwd,
    fmha_fwd,
    fmha_fwd_appendkv,
    fmha_fwd_splitkv,
    fmha_pagedkv_prefill,
)


class HcuCodegenTest(unittest.TestCase):
    modules = (
        fmha_fwd,
        fmha_bwd,
        fmha_fwd_appendkv,
        fmha_fwd_splitkv,
        fmha_pagedkv_prefill,
    )

    def test_paged_dispatch_allows_multiple_v_tiles(self):
        for target in ('gfx936', 'gfx938'):
            pool, kernels = fmha_pagedkv_prefill.get_fwd_blobs(
                [target], '', 4, [64, 128], 'simplified')
            self.assertTrue(kernels)
            self.assertIn('t.hdim_q <= 64 && t.hdim_v <= 64', pool.api)
            self.assertIn('t.hdim_q <= 128 && t.hdim_v <= 128', pool.api)

    def test_torch_receipt_emits_splitkv_for_both_dtypes(self):
        for target in ('gfx936', 'gfx938'):
            kernels = fmha_fwd_splitkv.get_fwd_splitkv_blobs(
                [target], '', 4, 'simplified', [64, 128])
            self.assertEqual({k.F_dtype for k in kernels}, {'fp16', 'bf16'})
            self.assertEqual({k.F_hdim for k in kernels}, {64, 128})
            self.assertTrue(all(k.F_pipeline.F_logits == 'f' for k in kernels))

    def test_torch_fwd_matches_wrapper_lse_and_mask_contract(self):
        for target in ('gfx936', 'gfx938'):
            _, kernels = fmha_fwd.get_fwd_blobs(
                [target], '', 4, [64, 128], 'simplified')
            contracts = {(k.F_pipeline.F_mask, k.F_pipeline.F_lse)
                         for k in kernels}
            self.assertTrue({('s_no', 't'), ('s_mask', 't')} <= contracts)

    def test_target_features_and_generic_fallback(self):
        for module in self.modules:
            for target in ('gfx936', 'gfx938'):
                for suffix in ('', ':sramecc+:xnack-'):
                    with self.subTest(module=module.__name__, target=target + suffix):
                        arch = module.get_factory(target + suffix).arch
                        self.assertEqual(arch.name, target)
                        self.assertEqual(arch.tag, get_hcu_arch_tag(target))
                        self.assertEqual(arch.preprocessor_check, f'defined(__{target}__)')
            self.assertEqual(module.get_factory('gfx942').arch.name, 'gfx9')
            self.assertEqual(module.get_factory('gfx1100').arch.name, 'gfx11')
            factories = get_factories_for_targets(
                ['gfx9', 'gfx936', 'gfx938', 'gfx936:xnack-'], module.get_factory)
            self.assertEqual([f.arch.name for f in factories], ['gfx936', 'gfx938', 'gfx9'])

    def test_generated_specializations_and_launches(self):
        for target in ('gfx936', 'gfx938'):
            # Receipt 2 includes main split-KV kernels as well as the combine pass.
            _, append = fmha_fwd_appendkv.get_fwd_appendkv_blobs([target], '', 2, 'simplified', [64])
            split = fmha_fwd_splitkv.get_fwd_splitkv_blobs([target], '', 2, 'simplified', [64])
            combine = fmha_fwd_splitkv.get_fwd_splitkv_combine_blobs([target], '', 2, [64])
            _, paged = fmha_pagedkv_prefill.get_fwd_blobs([target], '', 2, [128], 'simplified')
            for family, kernels in (('append', append), ('split', split), ('combine', combine), ('paged', paged)):
                with self.subTest(target=target, family=family):
                    self.assertTrue(kernels)
                    for kernel in kernels:
                        source = kernel.template
                        self.assertNotIn('ck_tile::gfx9_t', source)
                        self.assertIn(f'defined(__{target}__)', source)
                        self.assertIn(f', {get_hcu_arch_tag(target)}>', source)
                        self.assertIn('make_kernel<CK_TILE_MAX_THREAD_PER_BLOCK, kBlockPerCu>', source)
                        self.assertTrue(kernel.filename.endswith(f'_{target}.cpp'))


if __name__ == '__main__':
    unittest.main()
