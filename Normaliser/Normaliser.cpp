// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Normaliser.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

void CheckClip(float clip)
{
    if (!FiniteBits(clip) || !(clip > 0.f))
        throw std::invalid_argument("Normaliser clip must be finite and > 0");
}

} // namespace

Normaliser::Normaliser(std::vector<float> mean, std::vector<float> std, float clip)
    : mean_(std::move(mean)), std_(std::move(std)), clip_(clip)
{
}

Normaliser Normaliser::Fit(std::span<const float> x, size_t dim, float clip)
{
    CheckClip(clip);
    if (dim == 0)
        throw std::invalid_argument("Normaliser::Fit dim must be > 0");
    if (x.size() % dim != 0)
        throw std::invalid_argument("Normaliser::Fit x length must be a multiple of dim");
    const size_t count = x.size() / dim;
    if (count == 0)
        throw std::invalid_argument("Normaliser::Fit needs at least one row");

    std::vector<double> sum(dim, 0.0), sq(dim, 0.0);
    for (size_t i = 0; i < count; ++i)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            const double v = static_cast<double>(x[i * dim + d]);
            sum[d] += v;
            sq[d] += v * v;
        }
    }
    const double n = static_cast<double>(count);
    std::vector<float> mean(dim), stdv(dim);
    for (size_t d = 0; d < dim; ++d)
    {
        const double m = sum[d] / n;
        const double var = sq[d] / n - m * m;
        mean[d] = static_cast<float>(m);
        const float s = static_cast<float>(std::sqrt(var > 0.0 ? var : 0.0));
        stdv[d] = s > 1e-6f ? s : 1e-6f;
    }
    return Normaliser(std::move(mean), std::move(stdv), clip);
}

Normaliser Normaliser::FromState(std::span<const float> mean,
                                 std::span<const float> std,
                                 float clip)
{
    CheckClip(clip);
    if (mean.empty() || mean.size() != std.size())
        throw std::invalid_argument("Normaliser::FromState mean and std must be the same non-zero length");
    std::vector<float> m(mean.begin(), mean.end());
    std::vector<float> s(std.begin(), std.end());
    for (float& v : s)
        if (v < 1e-6f)
            v = 1e-6f;
    return Normaliser(std::move(m), std::move(s), clip);
}

void Normaliser::Apply(std::span<const float> x, std::span<float> dst) const
{
    const size_t dim = Dim();
    if (dim == 0)
        throw std::invalid_argument("Normaliser::Apply on an empty Normaliser");
    if (x.size() % dim != 0)
        throw std::invalid_argument("Normaliser::Apply x length must be a multiple of Dim()");
    if (dst.size() != x.size())
        throw std::invalid_argument("Normaliser::Apply dst must match x");
    const size_t count = x.size() / dim;
    const float inv = 1.f / clip_;
    for (size_t i = 0; i < count; ++i)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            float z = (x[i * dim + d] - mean_[d]) / std_[d];
            if (z > clip_)
                z = clip_;
            else if (z < -clip_)
                z = -clip_;
            dst[i * dim + d] = z * inv;
        }
    }
}
