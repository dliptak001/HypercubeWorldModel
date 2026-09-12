// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Metrics.h"
#include "VectorModel.h"

#include <algorithm>
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
    std::vector<float> lo(ad, -1.f), hi(ad, 1.f);
    if (vm.HasActionBounds() && vm.ActionLow().size() == ad)
    {
        const auto L = vm.ActionLow(), H = vm.ActionHigh();
        lo.assign(L.begin(), L.end());
        hi.assign(H.begin(), H.end());
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<float> p1(n * c), p2(n * c), za(c), act(ad);
    for (size_t i = 0; i < n; ++i)
    {
        std::span<const float> zi(z.data() + i * c, c);
        for (int k = 0; k < 2; ++k)
        {
            for (size_t j = 0; j < ad; ++j)
                act[j] = lo[j] + (hi[j] - lo[j]) * u(rng);
            vm.EncodeAction(act, za);
            float* dst = (k == 0 ? p1 : p2).data() + i * c;
            vm.Predict(zi, za, std::span<float>(dst, c));
        }
    }
    return ActionSensitivityFromPreds(p1, p2, mse);
}

namespace {

// Least-squares [Z 1] W ≈ Y. Column-pivoted modified Gram-Schmidt so a
// rank-deficient Z (typical action codes) still yields the OLS fit on
// the column span; unused columns get weight 0. W is z_dim × y_dim,
// b is the affine term.
void LeastSquaresWithBias(std::span<const float> z, std::span<const float> y,
                          size_t n, size_t z_dim, size_t y_dim,
                          std::vector<double>& w, std::vector<double>& b)
{
    const size_t p = z_dim + 1;
    std::vector<double> X(n * p);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < z_dim; ++j)
            X[i * p + j] = static_cast<double>(z[i * z_dim + j]);
        X[i * p + z_dim] = 1.0;
    }

    std::vector<double> energy(p, 0.0);
    for (size_t j = 0; j < p; ++j)
        for (size_t i = 0; i < n; ++i)
            energy[j] += X[i * p + j] * X[i * p + j];
    double max_energy = 0.0;
    for (size_t j = 0; j < p; ++j)
        if (energy[j] > max_energy)
            max_energy = energy[j];
    const double tol = 1e-12 * (max_energy > 0.0 ? max_energy : 1.0);

    std::vector<size_t> perm(p);
    for (size_t j = 0; j < p; ++j)
        perm[j] = j;
    std::vector<double> R(p * p, 0.0);
    size_t rank = 0;
    for (size_t k = 0; k < p; ++k)
    {
        size_t jmax = k;
        for (size_t j = k + 1; j < p; ++j)
            if (energy[j] > energy[jmax])
                jmax = j;
        if (energy[jmax] < tol)
            break;
        if (jmax != k)
        {
            for (size_t i = 0; i < n; ++i)
                std::swap(X[i * p + k], X[i * p + jmax]);
            std::swap(energy[k], energy[jmax]);
            std::swap(perm[k], perm[jmax]);
            for (size_t t = 0; t < k; ++t)
                std::swap(R[t * p + k], R[t * p + jmax]);
        }
        for (size_t t = 0; t < k; ++t)
        {
            double dot = 0.0;
            for (size_t i = 0; i < n; ++i)
                dot += X[i * p + t] * X[i * p + k];
            R[t * p + k] += dot;
            for (size_t i = 0; i < n; ++i)
                X[i * p + k] -= dot * X[i * p + t];
        }
        double nrm2 = 0.0;
        for (size_t i = 0; i < n; ++i)
            nrm2 += X[i * p + k] * X[i * p + k];
        const double nrm = std::sqrt(nrm2);
        if (!(nrm > 0.0) || nrm2 < tol)
            break;
        R[k * p + k] = nrm;
        for (size_t i = 0; i < n; ++i)
            X[i * p + k] /= nrm;
        for (size_t j = k + 1; j < p; ++j)
        {
            double dot = 0.0;
            for (size_t i = 0; i < n; ++i)
                dot += X[i * p + k] * X[i * p + j];
            R[k * p + j] = dot;
            for (size_t i = 0; i < n; ++i)
                X[i * p + j] -= dot * X[i * p + k];
            double e = 0.0;
            for (size_t i = 0; i < n; ++i)
                e += X[i * p + j] * X[i * p + j];
            energy[j] = e;
        }
        rank = k + 1;
    }

    w.assign(z_dim * y_dim, 0.0);
    b.assign(y_dim, 0.0);
    if (rank == 0)
        return;

    std::vector<double> qty(rank * y_dim, 0.0);
    for (size_t t = 0; t < rank; ++t)
        for (size_t k = 0; k < y_dim; ++k)
        {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i)
                s += X[i * p + t] * static_cast<double>(y[i * y_dim + k]);
            qty[t * y_dim + k] = s;
        }

    std::vector<double> wp(rank * y_dim, 0.0);
    for (size_t k = 0; k < y_dim; ++k)
        for (size_t t = rank; t-- > 0;)
        {
            double s = qty[t * y_dim + k];
            for (size_t j = t + 1; j < rank; ++j)
                s -= R[t * p + j] * wp[j * y_dim + k];
            wp[t * y_dim + k] = s / R[t * p + t];
        }

    for (size_t t = 0; t < rank; ++t)
    {
        const size_t col = perm[t];
        for (size_t k = 0; k < y_dim; ++k)
        {
            if (col == z_dim)
                b[k] = wp[t * y_dim + k];
            else
                w[col * y_dim + k] = wp[t * y_dim + k];
        }
    }
}

} // namespace

LinearR2 LinearR2On(std::span<const float> z, std::span<const float> y,
                    std::span<const float> z_val, std::span<const float> y_val,
                    size_t train_count, size_t val_count,
                    size_t z_dim, size_t y_dim)
{
    if (train_count == 0 || val_count == 0 || z_dim == 0 || y_dim == 0)
        throw std::invalid_argument("LinearR2On needs non-zero counts and dims");
    if (z.size() != train_count * z_dim || y.size() != train_count * y_dim)
        throw std::invalid_argument("LinearR2On train z/y length");
    if (z_val.size() != val_count * z_dim || y_val.size() != val_count * y_dim)
        throw std::invalid_argument("LinearR2On val z/y length");

    std::vector<double> w, b;
    LeastSquaresWithBias(z, y, train_count, z_dim, y_dim, w, b);

    std::vector<double> pred(val_count * y_dim, 0.0);
    for (size_t i = 0; i < val_count; ++i)
        for (size_t k = 0; k < y_dim; ++k)
        {
            double s = b[k];
            const float* zi = z_val.data() + i * z_dim;
            for (size_t j = 0; j < z_dim; ++j)
                s += w[j * y_dim + k] * static_cast<double>(zi[j]);
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
                                   size_t windows, size_t path_len, size_t code_size)
{
    if (windows == 0 || path_len < 2 || code_size == 0)
        throw std::invalid_argument("RolloutErrorFromCodes needs windows > 0, H+1 >= 2");
    const size_t n = windows * path_len * code_size;
    if (z_pred.size() != n || z_true.size() != n)
        throw std::invalid_argument("RolloutErrorFromCodes length must be windows * (H+1) * code");
    const size_t H = path_len - 1;
    RolloutError out;
    out.error.assign(H, 0.f);
    out.baseline.assign(H, 0.f);
    out.ratio.assign(H, 0.f);
    const double denom = static_cast<double>(windows * code_size);
    for (size_t h = 0; h < H; ++h)
    {
        double err = 0.0, base = 0.0;
        for (size_t w = 0; w < windows; ++w)
        {
            const float* pred = z_pred.data() + (w * path_len + (h + 1)) * code_size;
            const float* tru = z_true.data() + (w * path_len + (h + 1)) * code_size;
            const float* z0 = z_true.data() + w * path_len * code_size;
            for (size_t c = 0; c < code_size; ++c)
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
