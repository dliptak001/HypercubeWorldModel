// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Head.h"
#include "LCNTraining.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

float QuietNaN()
{
    return std::bit_cast<float>(0x7fc00000u);
}

size_t CubeDim(size_t n)
{
    if (n < 16 || (n & (n - 1)) != 0)
        throw std::invalid_argument(
            "Head code (or packed z||za) must be a power of two, at least 16");
    size_t d = 0;
    while ((size_t{1} << d) < n)
        ++d;
    if (d > 24)
        throw std::invalid_argument("Head packed cube dim must be <= 24");
    return d;
}

} // namespace

Head::Head()
{
    cfg_.sign = Sign::Cost;
    cfg_.seed = 1;
    cfg_.z_max = 0;
    cfg_.gather_span = 2;
    cfg_.tanh_last = false;
    cfg_.lr = 1e-2f;
    cfg_.lr_min_frac = 0.02f;
    cfg_.lr_decay_epochs = 0;
    cfg_.restore_best = true;
}

Head::Head(Config cfg) : cfg_(cfg)
{
    if (cfg_.gather_span < 2 || cfg_.gather_span > 6)
        throw std::invalid_argument("Head gather_span must be in [2, 6]");
    if (!FiniteBits(cfg_.lr) || !(cfg_.lr > 0.f))
        throw std::invalid_argument("Head lr must be finite and > 0");
    if (!FiniteBits(cfg_.lr_min_frac) || cfg_.lr_min_frac < 0.f || cfg_.lr_min_frac > 1.f)
        throw std::invalid_argument("Head lr_min_frac must be in [0, 1]");
}

Head::Head(Sign sign)
    : Head()
{
    cfg_.sign = sign;
}

Head::Head(const Head& o)
    : cfg_(o.cfg_), uses_za_(o.uses_za_), fitted_(o.fitted_), code_(o.code_)
{
    if (o.net_)
    {
        net_ = LCN::Create(o.net_->Config());
        net_->LoadWeights(o.net_->Weights());
        field_.assign(net_->N(), 0.f);
    }
}

Head& Head::operator=(const Head& o)
{
    if (this != &o)
    {
        cfg_ = o.cfg_;
        uses_za_ = o.uses_za_;
        fitted_ = o.fitted_;
        code_ = o.code_;
        net_.reset();
        field_.clear();
        if (o.net_)
        {
            net_ = LCN::Create(o.net_->Config());
            net_->LoadWeights(o.net_->Weights());
            field_.assign(net_->N(), 0.f);
        }
    }
    return *this;
}

Head::Head(Head&&) noexcept = default;
Head& Head::operator=(Head&&) noexcept = default;
Head::~Head() = default;

const LCN& Head::Net() const
{
    if (!net_)
        throw std::invalid_argument("Head::Net requires Fit");
    return *net_;
}

void Head::CheckZa(bool has_za) const
{
    if (uses_za_ && !has_za)
        throw std::invalid_argument("Head was fit with za; pass za");
    if (!uses_za_ && has_za)
        throw std::invalid_argument("Head was fit without za; omit za");
}

void Head::EnsureNet(size_t field_n)
{
    const size_t dim = CubeDim(field_n);
    net_ = LCN::Create(LCNConfig{.dim = dim,
                                 .seed = cfg_.seed,
                                 .z_max = cfg_.z_max,
                                 .gather_span = cfg_.gather_span,
                                 .tanh_last = cfg_.tanh_last});
    field_.assign(field_n, 0.f);
}

float Head::ForwardOne(const float* z, const float* za) const
{
    if (field_.size() != net_->N())
        field_.assign(net_->N(), 0.f);
    Pack(z, za, field_);
    net_->Forward(field_);
    return Readout();
}

void Head::Pack(const float* z, const float* za, std::span<float> field) const
{
    std::copy(z, z + code_, field.begin());
    if (uses_za_)
        std::copy(za, za + code_, field.begin() + static_cast<std::ptrdiff_t>(code_));
}

float Head::Readout() const
{
    return net_->Output()[0];
}

void Head::Fit(std::span<const float> z, size_t code_size, std::span<const float> y,
               int epochs, size_t batch, std::span<const float> za)
{
    if (code_size == 0)
        throw std::invalid_argument("Head::Fit code_size must be > 0");
    const size_t count = y.size();
    if (count == 0)
        throw std::invalid_argument("Head::Fit needs at least one row");
    if (z.size() != count * code_size)
        throw std::invalid_argument("Head::Fit z length must be count * code_size");
    if (epochs <= 0)
        throw std::invalid_argument("Head::Fit epochs must be > 0");
    if (batch == 0)
        throw std::invalid_argument("Head::Fit batch must be > 0");
    uses_za_ = !za.empty();
    if (uses_za_ && za.size() != count * code_size)
        throw std::invalid_argument("Head::Fit za length must be count * code_size");
    code_ = code_size;

    const size_t field_n = uses_za_ ? 2 * code_size : code_size;
    EnsureNet(field_n);

    LCNTrainingConfig tc;
    tc.lr = cfg_.lr;
    tc.lr_min_frac = cfg_.lr_min_frac;
    tc.lr_decay_epochs = cfg_.lr_decay_epochs;
    tc.restore_best = cfg_.restore_best;
    LCNTraining train(*net_, tc);

    std::vector<float> target(1);
    std::vector<size_t> idx(count);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(cfg_.seed);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        train.SetEpoch(epoch, epochs);
        std::shuffle(idx.begin(), idx.end(), rng);
        double epoch_loss = 0.0;
        int nloss = 0;
        for (size_t start = 0; start < count; start += batch)
        {
            train.ZeroGrad();
            const size_t end = std::min(start + batch, count);
            for (size_t t = start; t < end; ++t)
            {
                const size_t i = idx[t];
                Pack(z.data() + i * code_size,
                     uses_za_ ? za.data() + i * code_size : nullptr, field_);
                net_->Forward(field_);
                target[0] = y[i];
                epoch_loss += static_cast<double>(train.Loss(target));
                ++nloss;
                train.Backward();
            }
            train.Adam();
        }
        if (nloss)
            train.Observe(static_cast<float>(epoch_loss / nloss), epoch);
    }
    train.RestoreBest();
    fitted_ = true;
}

void Head::Predict(std::span<const float> z, std::span<float> dst,
                   std::span<const float> za) const
{
    if (!fitted_ || !net_)
        throw std::invalid_argument("Head::Predict requires Fit");
    if (code_ == 0 || z.size() % code_ != 0)
        throw std::invalid_argument("Head::Predict z length must be a multiple of code");
    const size_t count = z.size() / code_;
    if (dst.size() != count)
        throw std::invalid_argument("Head::Predict dst must have one value per row");
    CheckZa(!za.empty());
    if (uses_za_ && za.size() != count * code_)
        throw std::invalid_argument("Head::Predict za length must be count * code");

    for (size_t i = 0; i < count; ++i)
        dst[i] = ForwardOne(z.data() + i * code_,
                            uses_za_ ? za.data() + i * code_ : nullptr);
}

Head::Score Head::ScoreOn(std::span<const float> z, std::span<const float> y,
                          std::span<const float> za) const
{
    if (y.size() == 0)
        throw std::invalid_argument("Head::ScoreOn y must not be empty");
    std::vector<float> p(y.size());
    Predict(z, p, za);
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

void Head::PlanCost(std::span<const float> zs, std::span<float> out) const
{
    if (uses_za_)
        throw std::invalid_argument(
            "plan_cost scores view codes only; this Head was fit with za");
    if (!fitted_ || !net_)
        throw std::invalid_argument("Head::PlanCost requires Fit");
    const size_t batch = out.size();
    if (batch == 0 || code_ == 0 || zs.size() % (batch * code_) != 0)
        throw std::invalid_argument(
            "Head::PlanCost zs length must be batch * (H+1) * code");
    const size_t path_len = zs.size() / (batch * code_);
    if (path_len < 2)
        throw std::invalid_argument("Head::PlanCost needs H+1 >= 2");

    const size_t steps = path_len - 1;
    const float sgn = cfg_.sign == Sign::Reward ? -1.f : 1.f;
    for (size_t b = 0; b < batch; ++b)
    {
        double s = 0.0;
        for (size_t t = 0; t < steps; ++t)
            s += static_cast<double>(
                ForwardOne(zs.data() + (b * path_len + (t + 1)) * code_, nullptr));
        out[b] = sgn * static_cast<float>(s);
    }
}

Head Head::FromState(const Config& cfg, bool uses_za, size_t code_size,
                     std::span<const float> weights)
{
    Head h(cfg);
    h.uses_za_ = uses_za;
    h.code_ = code_size;
    const size_t field_n = uses_za ? 2 * code_size : code_size;
    h.EnsureNet(field_n);
    h.net_->LoadWeights(weights);
    h.fitted_ = true;
    return h;
}
