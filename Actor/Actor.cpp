// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Actor.h"
#include "LCNTraining.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

size_t CubeDim(size_t n)
{
    if (n < 16 || (n & (n - 1)) != 0)
        throw std::invalid_argument(
            "Actor code must be a power of two, at least 16");
    size_t d = 0;
    while ((size_t{1} << d) < n)
        ++d;
    if (d > 24)
        throw std::invalid_argument("Actor cube dim must be <= 24");
    return d;
}

} // namespace

Actor::Actor()
{
    cfg_.seed = 1;
    cfg_.z_max = 0;
    cfg_.gather_span = 2;
    cfg_.tanh_last = false;
    cfg_.lr = 1e-2f;
    cfg_.lr_min_frac = 0.02f;
    cfg_.lr_decay_epochs = 0;
    cfg_.restore_best = true;
}

Actor::Actor(Config cfg) : cfg_(cfg)
{
    if (cfg_.gather_span < 2 || cfg_.gather_span > 6)
        throw std::invalid_argument("Actor gather_span must be in [2, 6]");
    if (!FiniteBits(cfg_.lr) || !(cfg_.lr > 0.f))
        throw std::invalid_argument("Actor lr must be finite and > 0");
    if (!FiniteBits(cfg_.lr_min_frac) || cfg_.lr_min_frac < 0.f || cfg_.lr_min_frac > 1.f)
        throw std::invalid_argument("Actor lr_min_frac must be in [0, 1]");
}

Actor::Actor(const Actor& o)
    : cfg_(o.cfg_), fitted_(o.fitted_), code_(o.code_), act_(o.act_)
{
    if (o.net_)
    {
        net_ = LCN::Create(o.net_->Config());
        net_->LoadWeights(o.net_->Weights());
        field_.assign(net_->N(), 0.f);
    }
}

Actor& Actor::operator=(const Actor& o)
{
    if (this != &o)
    {
        Actor tmp(o);
        *this = std::move(tmp);
    }
    return *this;
}

Actor::Actor(Actor&&) noexcept = default;
Actor& Actor::operator=(Actor&&) noexcept = default;
Actor::~Actor() = default;

const LCN& Actor::Net() const
{
    if (!net_)
        throw std::invalid_argument("Actor::Net requires Fit");
    return *net_;
}

void Actor::EnsureNet(size_t field_n)
{
    const size_t dim = CubeDim(field_n);
    net_ = LCN::Create(LCNConfig{.dim = dim,
                                 .seed = cfg_.seed,
                                 .z_max = cfg_.z_max,
                                 .gather_span = cfg_.gather_span,
                                 .tanh_last = cfg_.tanh_last});
    field_.assign(field_n, 0.f);
}

void Actor::ForwardOne(const float* z, float* dst) const
{
    if (field_.size() != net_->N())
        field_.assign(net_->N(), 0.f);
    std::copy(z, z + code_, field_.begin());
    net_->Forward(field_);
    const auto& out = net_->Output();
    std::copy(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(act_), dst);
}

void Actor::Fit(std::span<const float> z, size_t code_size,
                std::span<const float> a, size_t act_dim,
                int epochs, size_t batch)
{
    Fit(z, code_size, a, act_dim, epochs, batch, FitOptions{});
}

void Actor::Fit(std::span<const float> z, size_t code_size,
                std::span<const float> a, size_t act_dim,
                int epochs, size_t batch, const FitOptions& opt)
{
    if (code_size == 0)
        throw std::invalid_argument("Actor::Fit code_size must be > 0");
    if (act_dim == 0 || act_dim > code_size)
        throw std::invalid_argument("Actor::Fit act_dim must be in [1, code_size]");
    if (a.size() % act_dim != 0)
        throw std::invalid_argument("Actor::Fit a length must be count * act_dim");
    const size_t count = a.size() / act_dim;
    if (count == 0)
        throw std::invalid_argument("Actor::Fit needs at least one row");
    if (z.size() != count * code_size)
        throw std::invalid_argument("Actor::Fit z length must be count * code_size");
    if (epochs <= 0)
        throw std::invalid_argument("Actor::Fit epochs must be > 0");
    if (batch == 0)
        throw std::invalid_argument("Actor::Fit batch must be > 0");
    if (opt.val_a.size() % act_dim != 0)
        throw std::invalid_argument("Actor::Fit val_a length must be val count * act_dim");
    const size_t val_count = opt.val_a.size() / act_dim;
    if (val_count == 0 && !opt.val_z.empty())
        throw std::invalid_argument("Actor::Fit val needs val_a");
    if (val_count && opt.val_z.size() != val_count * code_size)
        throw std::invalid_argument("Actor::Fit val_z length must be val count * code_size");

    const size_t dim = CubeDim(code_size);
    auto net = LCN::Create(LCNConfig{.dim = dim,
                                     .seed = cfg_.seed,
                                     .z_max = cfg_.z_max,
                                     .gather_span = cfg_.gather_span,
                                     .tanh_last = cfg_.tanh_last});
    std::vector<float> field(code_size, 0.f);

    LCNTrainingConfig tc;
    tc.lr = cfg_.lr;
    tc.lr_min_frac = cfg_.lr_min_frac;
    tc.lr_decay_epochs = cfg_.lr_decay_epochs;
    tc.restore_best = cfg_.restore_best;
    LCNTraining train(*net, tc);

    std::vector<float> target(act_dim);
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
                std::copy(z.data() + i * code_size,
                          z.data() + (i + 1) * code_size, field.begin());
                net->Forward(field);
                std::copy(a.data() + i * act_dim,
                          a.data() + (i + 1) * act_dim, target.begin());
                epoch_loss += static_cast<double>(train.Loss(target));
                ++nloss;
                train.Backward();
            }
            train.Adam();
        }
        const float train_loss =
            nloss ? static_cast<float>(epoch_loss / nloss) : 0.f;
        std::optional<float> val_loss;
        if (val_count)
        {
            // Forward only. The next training step runs its own Forward
            // before Loss, so this leaves no stale state behind.
            double sum = 0.0;
            for (size_t i = 0; i < val_count; ++i)
            {
                std::copy(opt.val_z.data() + i * code_size,
                          opt.val_z.data() + (i + 1) * code_size, field.begin());
                net->Forward(field);
                const auto& out = net->Output();
                for (size_t v = 0; v < act_dim; ++v)
                {
                    const double d = static_cast<double>(out[v]) -
                                     static_cast<double>(opt.val_a[i * act_dim + v]);
                    sum += 0.5 * d * d;
                }
            }
            val_loss = static_cast<float>(sum / static_cast<double>(val_count));
        }
        if (val_count)
            train.Observe(*val_loss, epoch);
        else if (nloss)
            train.Observe(train_loss, epoch);
        // net is a local until the end, so a throw here leaves *this as it was.
        if (opt.on_epoch)
            opt.on_epoch(epoch, epochs, train_loss, val_loss);
    }
    train.RestoreBest();
    net_ = std::move(net);
    field_ = std::move(field);
    code_ = code_size;
    act_ = act_dim;
    fitted_ = true;
}

void Actor::Predict(std::span<const float> z, std::span<float> dst) const
{
    if (!fitted_ || !net_)
        throw std::invalid_argument("Actor::Predict requires Fit");
    if (code_ == 0 || z.size() % code_ != 0)
        throw std::invalid_argument("Actor::Predict z length must be a multiple of code");
    const size_t count = z.size() / code_;
    if (dst.size() != count * act_)
        throw std::invalid_argument("Actor::Predict dst must be count * act_dim");

    for (size_t i = 0; i < count; ++i)
        ForwardOne(z.data() + i * code_, dst.data() + i * act_);
}

Actor::Score Actor::ScoreOn(std::span<const float> z, std::span<const float> a) const
{
    if (!fitted_ || !net_)
        throw std::invalid_argument("Actor::ScoreOn requires Fit");
    if (act_ == 0 || a.size() % act_ != 0)
        throw std::invalid_argument("Actor::ScoreOn a length must be count * act_dim");
    const size_t count = a.size() / act_;
    if (count == 0)
        throw std::invalid_argument("Actor::ScoreOn a must not be empty");
    std::vector<float> p(count * act_);
    Predict(z, p);
    double ymean = 0.0;
    for (float v : a)
        ymean += static_cast<double>(v);
    ymean /= static_cast<double>(a.size());
    double sse = 0.0, sst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double dy = static_cast<double>(p[i]) - static_cast<double>(a[i]);
        const double dt = static_cast<double>(a[i]) - ymean;
        sse += dy * dy;
        sst += dt * dt;
    }
    Score s;
    s.r2 = static_cast<float>(1.0 - sse / (sst > 1e-12 ? sst : 1e-12));
    return s;
}

Actor Actor::FromState(const Config& cfg, size_t code_size, size_t act_dim,
                       std::span<const float> weights)
{
    if (act_dim == 0 || act_dim > code_size)
        throw std::invalid_argument("Actor::FromState act_dim must be in [1, code_size]");
    Actor h(cfg);
    h.EnsureNet(code_size);
    h.net_->LoadWeights(weights);
    h.code_ = code_size;
    h.act_ = act_dim;
    h.fitted_ = true;
    return h;
}
