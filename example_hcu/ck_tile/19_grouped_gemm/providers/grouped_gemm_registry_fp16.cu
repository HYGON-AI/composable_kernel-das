// 实例化条件（编译期）：构建目标包含 gfx936 或 gfx938，并启用
// CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY。REGISTRY_ONLY 只生成 FP16
// device-args registry/launcher，不重复生成旧 host-descriptor C ABI。
// 运行时由 pure selector 在 gfx936 V3、gfx938 MLS 和通用 V4 预编译候选中选择；
// 不满足大块/对齐条件时仍落到 V4 预编译 fallback。
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_ONLY
#include "instances/grouped_gemm_fp16.cpp"
