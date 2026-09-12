// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Head.h"
#include "Solver.h"

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

float QuietNaN()
{
    return std::bit_cast<float>(0x7fc00000u);
}

size_t CodeFeatureCount(Head::Kind kind, size_t code)
{
    if (kind == Head::Kind::Linear)
        return code;
    return code + code * (code + 1) / 2;
}

void CodeFeaturesRow(Head::Kind kind, const float* z, size_t code, float* out)
{
    if (kind == Head::Kind::Linear)
    {
        for (size_t i = 0; i < code; ++i)
            out[i] = z[i];
        return;
    }
    for (size_t i = 0; i < code; ++i)
        out[i] = z[i];
    size_t p = code;
    for (size_t i = 0; i < code; ++i)
        for (size_t j = i; j < code; ++j)
            out[p++] = z[i] * z[j];
}

} // namespace

Head::Head(Kind kind, float ridge, Sign sign)
    : kind_(kind), sign_(sign), ridge_(ridge)
{
    if (!FiniteBits(ridge) || !(ridge > 0.f))
        throw std::invalid_argument("Head ridge must be finite and > 0");
}

size_t Head::CodeFeatures() const
{
    return CodeFeatureCount(kind_, code_);
}

void Head::CheckZa(bool has_za) const
{
    if (uses_za_ && !has_za)
        throw std::invalid_argument("Head was fit with za; pass za");
    if (!uses_za_ && has_za)
        throw std::invalid_argument("Head was fit without za; omit za");
}

void Head::Features(std::span<const float> z, std::span<const float> za,
                    size_t count, std::vector<float>& out) const
{
    const size_t cf = CodeFeatures();
    const size_t nf = uses_za_ ? 2 * cf : cf;
    out.resize(count * nf);
    for (size_t i = 0; i < count; ++i)
    {
        CodeFeaturesRow(kind_, z.data() + i * code_, code_, out.data() + i * nf);
        if (uses_za_)
            CodeFeaturesRow(kind_, za.data() + i * code_, code_,
                            out.data() + i * nf + cf);
    }
}

void Head::Fit(std::span<const float> z, size_t code, std::span<const float> y,
               size_t count, std::span<const float> za)
{
    if (code == 0)
        throw std::invalid_argument("Head::Fit code must be > 0");
    if (count == 0)
        throw std::invalid_argument("Head::Fit needs at least one row");
    if (z.size() != count * code)
        throw std::invalid_argument("Head::Fit z length must be count * code");
    if (y.size() != count)
        throw std::invalid_argument("Head::Fit y length must be count");
    uses_za_ = !za.empty();
    if (uses_za_ && za.size() != count * code)
        throw std::invalid_argument("Head::Fit za length must be count * code");
    code_ = code;

    std::vector<float> f32;
    Features(z, za, count, f32);
    const size_t nf = f32.size() / count;

    std::vector<double> f(f32.size());
    for (size_t i = 0; i < f32.size(); ++i)
        f[i] = static_cast<double>(f32[i]);

    mu_.assign(nf, 0.f);
    sd_.assign(nf, 0.f);
    std::vector<double> mu(nf, 0.0), sd(nf, 0.0);
    for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < nf; ++j)
            mu[j] += f[i * nf + j];
    const double n = static_cast<double>(count);
    for (size_t j = 0; j < nf; ++j)
        mu[j] /= n;
    for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < nf; ++j)
        {
            const double d = f[i * nf + j] - mu[j];
            sd[j] += d * d;
        }
    for (size_t j = 0; j < nf; ++j)
    {
        sd[j] = std::sqrt(sd[j] / n);
        if (sd[j] < 1e-8)
            sd[j] = 1e-8;
        mu_[j] = static_cast<float>(mu[j]);
        sd_[j] = static_cast<float>(sd[j]);
    }
    for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < nf; ++j)
            f[i * nf + j] = (f[i * nf + j] - mu[j]) / sd[j];

    double ymean = 0.0;
    for (size_t i = 0; i < count; ++i)
        ymean += static_cast<double>(y[i]);
    ymean /= n;
    b_ = static_cast<float>(ymean);

    std::vector<double> A(nf * nf, 0.0), rhs(nf, 0.0);
    for (size_t i = 0; i < count; ++i)
    {
        const double yc = static_cast<double>(y[i]) - ymean;
        const double* fi = f.data() + i * nf;
        for (size_t a = 0; a < nf; ++a)
        {
            rhs[a] += fi[a] * yc;
            for (size_t b = 0; b < nf; ++b)
                A[a * nf + b] += fi[a] * fi[b];
        }
    }
    const double lam = static_cast<double>(ridge_) * n;
    for (size_t a = 0; a < nf; ++a)
        A[a * nf + a] += lam;

    DenseSolve(A, nf, rhs, 1);
    w_.resize(nf);
    for (size_t j = 0; j < nf; ++j)
        w_[j] = static_cast<float>(rhs[j]);
}

void Head::Apply(std::span<const float> z, std::span<float> dst,
                 std::span<const float> za) const
{
    if (w_.empty())
        throw std::invalid_argument("Head::Apply requires Fit");
    if (code_ == 0 || z.size() % code_ != 0)
        throw std::invalid_argument("Head::Apply z length must be a multiple of code");
    const size_t count = z.size() / code_;
    if (dst.size() != count)
        throw std::invalid_argument("Head::Apply dst must have one value per row");
    CheckZa(!za.empty());
    if (uses_za_ && za.size() != count * code_)
        throw std::invalid_argument("Head::Apply za length must be count * code");

    std::vector<float> f;
    Features(z, za, count, f);
    const size_t nf = w_.size();
    for (size_t i = 0; i < count; ++i)
    {
        double s = static_cast<double>(b_);
        const float* fi = f.data() + i * nf;
        for (size_t j = 0; j < nf; ++j)
            s += static_cast<double>((fi[j] - mu_[j]) / sd_[j]) *
                 static_cast<double>(w_[j]);
        dst[i] = static_cast<float>(s);
    }
}

Head::Score Head::ScoreOn(std::span<const float> z, std::span<const float> y,
                          std::span<const float> za) const
{
    if (y.size() == 0)
        throw std::invalid_argument("Head::ScoreOn y must not be empty");
    std::vector<float> p(y.size());
    Apply(z, p, za);
    double ymean = 0.0;
    for (float v : y)
        ymean += static_cast<double>(v);
    ymean /= static_cast<double>(y.size());
    double sse = 0.0, sst = 0.0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        const double dy = static_cast<double>(p[i]) - static_cast<double>(y[i]);
        const double dt = static_cast<double>(y[i]) - ymean;
        sse += dy * dy;
        sst += dt * dt;
    }
    Score s;
    s.r2 = static_cast<float>(1.0 - sse / (sst > 1e-12 ? sst : 1e-12));

    bool binary = true;
    int pos = 0, neg = 0;
    for (float v : y)
    {
        if (v == 0.f)
            ++neg;
        else if (v == 1.f)
            ++pos;
        else
            binary = false;
    }
    s.positives = pos;
    if (!(binary && pos && neg))
    {
        s.auc = QuietNaN();
        s.has_auc = false;
        return s;
    }
    std::vector<float> pv, nv;
    for (size_t i = 0; i < y.size(); ++i)
    {
        if (y[i] == 1.f)
            pv.push_back(p[i]);
        else
            nv.push_back(p[i]);
    }
    double wins = 0.0;
    for (float a : pv)
        for (float b : nv)
        {
            if (a > b)
                wins += 1.0;
            else if (a == b)
                wins += 0.5;
        }
    s.auc = static_cast<float>(wins / (static_cast<double>(pv.size()) *
                                       static_cast<double>(nv.size())));
    s.has_auc = true;
    return s;
}

void Head::PlanCost(std::span<const float> zs, size_t batch, size_t h1,
                    std::span<float> out) const
{
    if (uses_za_)
        throw std::invalid_argument(
            "plan_cost scores view codes only; this Head was fit with za");
    if (w_.empty())
        throw std::invalid_argument("Head::PlanCost requires Fit");
    if (batch == 0 || h1 < 2)
        throw std::invalid_argument("Head::PlanCost needs batch > 0 and H+1 >= 2");
    if (zs.size() != batch * h1 * code_)
        throw std::invalid_argument("Head::PlanCost zs length must be batch * (H+1) * code");
    if (out.size() != batch)
        throw std::invalid_argument("Head::PlanCost out must be batch long");

    const size_t steps = h1 - 1;
    std::vector<float> z(batch * steps * code_);
    for (size_t b = 0; b < batch; ++b)
        for (size_t t = 0; t < steps; ++t)
            for (size_t c = 0; c < code_; ++c)
                z[(b * steps + t) * code_ + c] =
                    zs[(b * h1 + (t + 1)) * code_ + c];
    std::vector<float> pred(batch * steps);
    Apply(z, pred);
    const float sgn = sign_ == Sign::Reward ? -1.f : 1.f;
    for (size_t b = 0; b < batch; ++b)
    {
        double s = 0.0;
        for (size_t t = 0; t < steps; ++t)
            s += static_cast<double>(pred[b * steps + t]);
        out[b] = sgn * static_cast<float>(s);
    }
}

Head Head::FromState(Kind kind, Sign sign, float ridge, bool uses_za,
                     size_t code, std::span<const float> w, float b,
                     std::span<const float> mu, std::span<const float> sd)
{
    Head h(kind, ridge, sign);
    if (w.empty() || w.size() != mu.size() || w.size() != sd.size())
        throw std::invalid_argument("Head::FromState w, mu, sd must be the same non-zero length");
    const size_t cf = CodeFeatureCount(kind, code);
    const size_t expect = uses_za ? 2 * cf : cf;
    if (w.size() != expect)
        throw std::invalid_argument("Head::FromState feature count does not match kind/code/za");
    h.uses_za_ = uses_za;
    h.code_ = code;
    h.w_.assign(w.begin(), w.end());
    h.mu_.assign(mu.begin(), mu.end());
    h.sd_.assign(sd.begin(), sd.end());
    h.b_ = b;
    return h;
}
