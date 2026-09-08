#pragma once

#include <cstdint>

// Common enumeration describing the fused quantization strategies supported
// by fused MoE kernels. The numeric values mirror the legacy integer-based
// configuration knobs so existing serialized configs remain compatible.
enum class fused_quant_mode : int
{
    none = 0,
    smooth_dynamic = 1,
    int8_w8a16 = 2,
    int4_w4a16 = 3,
    int8_w8a8_block = 4,
    int4_w4a8_block = 5,
    fp8_w8a8_block = 6,
    fp8_w8a8_channel = 7,
    dynamic_quant = 10
};

constexpr fused_quant_mode fused_quant_mode_from_value(
    int value, fused_quant_mode fallback = fused_quant_mode::none)
{
    switch(value)
    {
    case 0: return fused_quant_mode::none;
    case 1: return fused_quant_mode::smooth_dynamic;
    case 2: return fused_quant_mode::int8_w8a16;
    case 3: return fused_quant_mode::int4_w4a16;
    case 4: return fused_quant_mode::int8_w8a8_block;
    case 5: return fused_quant_mode::int4_w4a8_block;
    case 6: return fused_quant_mode::fp8_w8a8_block;
    case 7: return fused_quant_mode::fp8_w8a8_channel;
    case 10: return fused_quant_mode::dynamic_quant;
    default: return fallback;
    }
}

constexpr int fused_quant_mode_to_value(fused_quant_mode mode)
{
    return static_cast<int>(mode);
}

constexpr const char* fused_quant_mode_to_string(fused_quant_mode mode)
{
    switch(mode)
    {
    case fused_quant_mode::none: return "none";
    case fused_quant_mode::smooth_dynamic: return "smooth_dynamic";
    case fused_quant_mode::int8_w8a16: return "int8_w8a16";
    case fused_quant_mode::int4_w4a16: return "int4_w4a16";
    case fused_quant_mode::int8_w8a8_block: return "int8_w8a8_block";
    case fused_quant_mode::int4_w4a8_block: return "int4_w4a8_block";
    case fused_quant_mode::fp8_w8a8_block: return "fp8_w8a8_block";
    case fused_quant_mode::fp8_w8a8_channel: return "fp8_w8a8_channel";
    case fused_quant_mode::dynamic_quant: return "dynamic_quant";
    default: return "unknown";
    }
}
