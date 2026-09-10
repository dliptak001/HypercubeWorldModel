#pragma once

#include "Encoder.h"

#include <cstdio>
#include <cstdint>

// Shared run banners so CompressionTest and JepaEncoderTest logs line up.

inline void PrintEncoderBanner(const char* name, const EncoderConfig& c, size_t n,
                               float sr_post, size_t k, size_t sub)
{
    std::printf("%s%senc  dim=%zu N=%zu k=%zu sub=%zu  seed=%llu ic_seed=%llu  "
                "SR=%.6g SR_post=%.4g leak=%.6g in_scale=%.6g M=%zu T=%zu\n",
                name && name[0] ? name : "",
                name && name[0] ? ": " : "",
                c.dim, n, k, sub,
                static_cast<unsigned long long>(c.seed),
                static_cast<unsigned long long>(c.ic_seed),
                static_cast<double>(c.spectral_radius),
                static_cast<double>(sr_post),
                static_cast<double>(c.leak_rate),
                static_cast<double>(c.input_scaling),
                c.history_depth, c.passes);
    std::fflush(stdout);
}

inline void PrintEncoderBanner(const char* name, const Encoder& enc,
                               size_t k, size_t sub)
{
    PrintEncoderBanner(name, enc.Config(), enc.Size(),
                       enc.RealizedSpectralRadius(), k, sub);
}

inline void PrintSineBanner(const char* name, int terms,
                            float cycles_min, float cycles_max,
                            float amp_min, float amp_max, uint64_t data_seed)
{
    std::printf("%s%ssine terms=%d cycles=[%.4g,%.4g] amp=[%.4g,%.4g] "
                "data_seed=%llu\n",
                name && name[0] ? name : "",
                name && name[0] ? ": " : "",
                terms,
                static_cast<double>(cycles_min),
                static_cast<double>(cycles_max),
                static_cast<double>(amp_min),
                static_cast<double>(amp_max),
                static_cast<unsigned long long>(data_seed));
    std::fflush(stdout);
}
