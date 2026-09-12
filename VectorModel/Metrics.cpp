// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Metrics.h"
#include "Solver.h"
#include "VectorModel.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

float QuietNaN()
{
    return std::bit_cast<float>(0x7fc00000u);
}

} // namespace

float NoChangeMse(std::span<const float> z, std::span<const float> zn)
{
    if (z.size() != zn.size() || z.empty())
        throw std::invalid_argument("NoChangeMse z and zn must be the same non-zero length");
    double s = 0.0;
    for (size_t i = 0; i < z.size(); ++i)
    {
        const double d = static_cast<double>(zn[i]) - static_cast<double>(z[i]);
        s += d * d;
    }
    return static_cast<float>(s / static_cast<double>(z.size()));
}

float OneStepRatio(float mse, std::span<const float> z, std::span<const float> zn)
{
    const float base = NoChangeMse(z, zn);
    if (!(base > 0.f))
        return QuietNaN();
    return mse / base;
}

ActionSensitivity ActionSensitivityFromPreds(std::span<const float> pred1,
                                             std::span<const float> pred2,
                                             float mse)
{
    if (pred1.size() != pred2.size() || pred1.empty())
        throw std::invalid_argument(
            "ActionSensitivityFromPreds predictions must be the same non-zero length");
    double s = 0.0;
    for (size_t i = 0; i < pred1.size(); ++i)
    {
        const double d = static_cast<double>(pred1[i]) - static_cast<double>(pred2[i]);
        s += d * d;
    }
    ActionSensitivity out;
    out.div = static_cast<float>(s / static_cast<double>(pred1.size()));
    out.mse = mse;
    out.act = mse > 0.f ? out.div / mse : QuietNaN();
    return out;
}

ActionSensitivity MeasureActionSensitivity(VectorModel& vm, std::span<const float> z,
                                           size_t count, float mse, uint64_t seed,
                                           size_t n)
{
    const size_t c = vm.CodeSize();
    if (count == 0 || z.size() != count * c)
        throw std::invalid_argument("MeasureActionSensitivity z must be count * code");
    if (vm.ActDim() == 0)
        throw std::invalid_argument("MeasureActionSensitivity needs act_dim");
    n = n < count ? n : count;
    const size_t ad = vm.ActDim();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> p1(n * c), p2(n * c), za(c), act(ad);
    for (size_t i = 0; i < n; ++i)
    {
        std::span<const float> zi(z.data() + i * c, c);
        for (int k = 0; k < 2; ++k)
        {
            for (size_t j = 0; j < ad; ++j)
                act[j] = u(rng);
            vm.EncodeAction(act, za);
            const float* hat = vm.Predict(zi, za);
            float* dst = (k == 0 ? p1 : p2).data() + i * c;
            for (size_t j = 0; j < c; ++j)
                dst[j] = hat[j];
        }
    }
    return ActionSensitivityFromPreds(p1, p2, mse);
}

LinearR2 LinearR2On(std::span<const float> z, std::span<const float> y,
                    std::span<const float> z_val, std::span<const float> y_val,
                    size_t train_count, size_t val_count,
                    size_t z_dim, size_t y_dim, float ridge)
{
    if (train_count == 0 || val_count == 0 || z_dim == 0 || y_dim == 0)
        throw std::invalid_argument("LinearR2On needs non-zero counts and dims");
    if (z.size() != train_count * z_dim || y.size() != train_count * y_dim)
        throw std::invalid_argument("LinearR2On train z/y length");
    if (z_val.size() != val_count * z_dim || y_val.size() != val_count * y_dim)
        throw std::invalid_argument("LinearR2On val z/y length");

    const size_t p = z_dim + 1;
    std::vector<double> A(p * p, 0.0), B(p * y_dim, 0.0);
    for (size_t i = 0; i < train_count; ++i)
    {
        for (size_t a = 0; a < p; ++a)
        {
            const double za = a < z_dim ? static_cast<double>(z[i * z_dim + a]) : 1.0;
            for (size_t b = 0; b < p; ++b)
            {
                const double zb = b < z_dim ? static_cast<double>(z[i * z_dim + b]) : 1.0;
                A[a * p + b] += za * zb;
            }
            for (size_t k = 0; k < y_dim; ++k)
                B[a * y_dim + k] += za * static_cast<double>(y[i * y_dim + k]);
        }
    }
    for (size_t a = 0; a < p; ++a)
        A[a * p + a] += static_cast<double>(ridge);

    DenseSolve(A, p, B, y_dim);

    std::vector<double> pred(val_count * y_dim, 0.0);
    for (size_t i = 0; i < val_count; ++i)
        for (size_t k = 0; k < y_dim; ++k)
        {
            double s = 0.0;
            for (size_t a = 0; a < p; ++a)
            {
                const double za = a < z_dim ? static_cast<double>(z_val[i * z_dim + a]) : 1.0;
                s += za * B[a * y_dim + k];
            }
            pred[i * y_dim + k] = s;
        }

    std::vector<double> ymean(y_dim, 0.0);
    for (size_t i = 0; i < val_count; ++i)
        for (size_t k = 0; k < y_dim; ++k)
            ymean[k] += static_cast<double>(y_val[i * y_dim + k]);
    for (size_t k = 0; k < y_dim; ++k)
        ymean[k] /= static_cast<double>(val_count);

    LinearR2 out;
    out.r2.resize(y_dim);
    double sum = 0.0;
    float mn = 1.f;
    for (size_t k = 0; k < y_dim; ++k)
    {
        double sse = 0.0, sst = 0.0;
        for (size_t i = 0; i < val_count; ++i)
        {
            const double yt = static_cast<double>(y_val[i * y_dim + k]);
            const double d = pred[i * y_dim + k] - yt;
            const double t = yt - ymean[k];
            sse += d * d;
            sst += t * t;
        }
        const float r = static_cast<float>(1.0 - sse / (sst > 1e-12 ? sst : 1e-12));
        out.r2[k] = r;
        sum += static_cast<double>(r);
        if (k == 0 || r < mn)
            mn = r;
    }
    out.min = mn;
    out.mean = static_cast<float>(sum / static_cast<double>(y_dim));
    return out;
}

RolloutError RolloutErrorFromCodes(std::span<const float> z_pred,
                                   std::span<const float> z_true,
                                   size_t windows, size_t h1, size_t code)
{
    if (windows == 0 || h1 < 2 || code == 0)
        throw std::invalid_argument("RolloutErrorFromCodes needs windows > 0, H+1 >= 2");
    const size_t n = windows * h1 * code;
    if (z_pred.size() != n || z_true.size() != n)
        throw std::invalid_argument("RolloutErrorFromCodes length must be windows * (H+1) * code");
    const size_t H = h1 - 1;
    RolloutError out;
    out.error.assign(H, 0.f);
    out.baseline.assign(H, 0.f);
    out.ratio.assign(H, 0.f);
    const double denom = static_cast<double>(windows * code);
    for (size_t h = 0; h < H; ++h)
    {
        double err = 0.0, base = 0.0;
        for (size_t w = 0; w < windows; ++w)
        {
            const float* pred = z_pred.data() + (w * h1 + (h + 1)) * code;
            const float* tru = z_true.data() + (w * h1 + (h + 1)) * code;
            const float* z0 = z_true.data() + w * h1 * code;
            for (size_t c = 0; c < code; ++c)
            {
                const double de = static_cast<double>(pred[c]) - static_cast<double>(tru[c]);
                const double db = static_cast<double>(z0[c]) - static_cast<double>(tru[c]);
                err += de * de;
                base += db * db;
            }
        }
        out.error[h] = static_cast<float>(err / denom);
        out.baseline[h] = static_cast<float>(base / denom);
        const float b = out.baseline[h];
        out.ratio[h] = b > 0.f ? out.error[h] / b : QuietNaN();
    }
    return out;
}
