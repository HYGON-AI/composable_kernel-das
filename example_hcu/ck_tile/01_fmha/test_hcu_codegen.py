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

    def test_bwd_allows_mask_padding_and_dropout_combos(self):
        # BWD 支持 causal mask、任意 seqlen（spad1d）、head_dim<=128（dpad/dvpad
        # 档位 0/8/1 与 FWD 对齐）、非零 dropout（load 型，掩码来自 randval 缓冲）。
        for target in ('gfx936', 'gfx938'):
            pool, _, kernels, _ = fmha_bwd.get_bwd_blobs([target], '', 3, 'simplified', [64, 128])
            self.assertTrue(kernels)
            masks = {k.F_mask for k in kernels}
            dropouts = {k.F_dropout for k in kernels}
            pads = {(k.F_dpad, k.F_dvpad) for k in kernels}
            self.assertTrue({'s_no', 's_mask'} <= masks)
            self.assertTrue({'no', 'dropout_ld', 'dropout_ld_storerandval'} <= dropouts)
            self.assertTrue({(0, 0), (8, 8), (1, 1)} <= pads)

            api = pool.api
            # head_dim 任意值（%1 档）与 %8 档分派存在
            self.assertIn('t.hdim_q % 1 == 0', api)
            self.assertIn('t.hdim_q % 8 == 0', api)
            # seqlen_q 非 64 对齐（1D kernel padding 档）分派存在
            self.assertIn('true /*t.seqlen_q % 64 != 0*/', api)
            # causal mask 分派（simplified: 非 no_mask 即走 s_mask kernel）
            self.assertIn('t.mask_type != mask_enum::no_mask', api)
            # load 型 dropout 分派（含 storerandval 校验档）
            self.assertIn('ck_tile::BlockDropoutBwdLoadRandval<false>', api)
            self.assertIn('ck_tile::BlockDropoutBwdLoadRandval<true>', api)

    def test_integration_receipts_replay_without_load_candidates(self):
        torch_filter = '*psdv*@*psd*@*_pd1dv1*_ntrload*'
        for target in ('gfx936', 'gfx938'):
            for receipt in (2, 4, 300, 600, 700):
                with self.subTest(target=target, receipt=receipt):
                    pool, _, kernels, _ = fmha_bwd.get_bwd_blobs(
                        [target], torch_filter if receipt == 4 else '',
                        receipt, 'simplified', [64, 128])
                    expected_dtypes = {'fp16', 'bf16'}
                    if receipt == 4:
                        expected_dtypes.add('bf16_fp32kv')
                    self.assertEqual({k.F_dtype for k in kernels}, expected_dtypes)
                    self.assertEqual({k.F_hdim for k in kernels}, {64, 128})
                    policies = {k.F_dropout for k in kernels}
                    self.assertTrue({'no', 'dropout_rp'} <= policies)
                    self.assertFalse(any(name.startswith('dropout_ld') for name in policies))
                    self.assertIn('ck_tile::BlockDropoutBwdReplayHcu<false>', pool.api)
                    if receipt == 600:
                        self.assertIn('dropout_rp_storerandval', policies)
            _, _, group, _ = fmha_bwd.get_bwd_blobs(
                [target], '', 400, 'simplified', [64, 128])
            self.assertFalse(group)

    def test_generic_architectures_retain_replay_policies(self):
        for target in ('gfx942', 'gfx950', 'gfx1100', 'gfx1201', 'gfx1250'):
            with self.subTest(target=target):
                pool, _, kernels, _ = fmha_bwd.get_bwd_blobs(
                    [target], '', 4, 'simplified', [64, 128])
                self.assertEqual({k.F_dropout for k in kernels}, {'no', 'dropout_wg16'})
                self.assertNotIn('BlockDropoutBwdLoadRandval', pool.api)
                self.assertNotIn('BlockDropoutBwdReplayHcu', pool.api)

    def test_fp32_kv_gradients_are_an_explicit_hcu_torch_variant(self):
        for target in ('gfx936', 'gfx938', 'gfx946', 'gfx942', 'gfx1100'):
            for receipt in (3, 4, 600):
                with self.subTest(target=target, receipt=receipt):
                    pool, _, kernels, _ = fmha_bwd.get_bwd_blobs(
                        [target], '*psdv*@*psd*@*_pd1dv1*_ntrload*',
                        receipt, 'simplified', [64, 128])
                    variants = [k for k in kernels if k.F_dtype == 'bf16_fp32kv']
                    expected = receipt == 4 and target in ('gfx936', 'gfx938')
                    self.assertEqual(bool(variants), expected)
                    if expected:
                        self.assertEqual(len(variants), 8)
                        self.assertIn('t.data_type.compare("bf16_fp32kv") == 0', pool.api)
                        self.assertTrue(all('FmhaBwdBf16Fp32KV' in k.template for k in variants))
                        self.assertTrue(any(k.F_dtype == 'bf16' for k in kernels))

    def test_gfx946_does_not_inherit_unverified_replay_mapping(self):
        _, _, kernels, _ = fmha_bwd.get_bwd_blobs(
            ['gfx946'], '', 3, 'simplified', [64, 128])
        self.assertEqual({k.F_dropout for k in kernels},
                         {'no', 'dropout_ld', 'dropout_ld_storerandval'})
        _, _, kernels, _ = fmha_bwd.get_bwd_blobs(
            ['gfx946'], '', 4, 'simplified', [64, 128])
        self.assertEqual({k.F_dropout for k in kernels}, {'no'})

    def test_hcu_fwd_skpad_and_dropout_pipelines(self):
        # FWD 支持任意 seqlen_k（skpad 档）与非零 dropout（逻辑 philox）。
        for target in ('gfx936', 'gfx938'):
            _, kernels = fmha_fwd.get_fwd_blobs([target], '', 3, [64, 128], 'simplified')
            combos = {(k.F_pipeline.F_skpad, k.F_pipeline.F_dropout) for k in kernels}
            self.assertTrue({('f', 'f'), ('t', 'f'), ('f', 't'), ('t', 't')} <= combos)

    def test_fwd_api_dispatch_skpad_and_dropout(self):
        import tempfile
        from pathlib import Path
        for target in ('gfx936', 'gfx938'):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                fmha_fwd.write_blobs([target], out, '', 3, [64, 128], 'simplified')
                api = (out / 'fmha_fwd_api.cpp').read_text()
                # seqlen_k % 64 快速档与非对齐回退档
                self.assertIn('a.seqlen_k % 64 == 0', api)
                self.assertIn('a.cu_seqlen_k_ptr != nullptr', api)
                # dropout 开/关互补档
                self.assertIn('t.has_dropout == true', api)
                self.assertIn('t.has_dropout == false', api)

    def test_bwd_padding_dispatch_order_prefers_fast_paths(self):
        # 严格对齐档在前（先 %64、再 %8、最后 %1），保证快路径优先命中。
        pool, _, _, _ = fmha_bwd.get_bwd_blobs(['gfx936'], '', 3, 'simplified', [64, 128])
        api = pool.api
        self.assertLess(api.index('t.hdim_q % 64 == 0'), api.index('t.hdim_q % 8 == 0'))
        self.assertLess(api.index('t.hdim_q % 8 == 0'), api.index('t.hdim_q % 1 == 0'))

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
