// 实例化条件（编译期）：本文件由 CK provider manifest 纳入 gfx936/gfx938
// device-args 构建，生成 BW-family FP16 的 NT/NN/TN 预编译入口。
// 当前 selector 仅在 gfx936、FP16、NT/NN、effective-M 达到 1920（device
// lengths）或 2048（common）、N/K>=2048、N%256==0、K%64==0 时选择
// bw_family_selected；其余已编译入口只用于 ABI 完整性或显式调用。
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY
#include "instances/grouped_gemm_gfx936_fp16.cpp"
