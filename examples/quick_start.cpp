// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// The quick start from docs/CPP_SDK.md. A point on a plane moves by a
// bounded velocity action. The view is its position, the action is the
// velocity, and the WorldModel learns to predict the next view code.

#include "WorldModel.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

namespace {

constexpr size_t kDim = 6;       // view cube; N = 2^dim
constexpr size_t kK = 5;         // kept face; 2^k code, k in [5, dim)
constexpr float kStep = 0.1f;    // position += kStep * velocity, clamped to [-1, 1]
constexpr int kTrain = 512, kVal = 128, kTest = 128;
constexpr int kEpochs = 200, kBatch = 16;

// obs and next are a position (x, y); act is a velocity (vx, vy) in [-1, 1].
struct Transition
{
    std::vector<float> obs, act, next;
};

std::vector<Transition> Draw(int count, std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<Transition> out(count);
    for (Transition& t : out)
    {
        t.obs = {u(rng), u(rng)};
        t.act = {u(rng), u(rng)};
        t.next = {std::clamp(t.obs[0] + kStep * t.act[0], -1.f, 1.f),
                  std::clamp(t.obs[1] + kStep * t.act[1], -1.f, 1.f)};
    }
    return out;
}

// Codes for one split: E(obs), E(a), E(next) per transition, laid end to end.
struct Codes
{
    std::vector<float> z, za, next;
};

Codes EncodeAll(WorldModel& wm, const std::vector<Transition>& data)
{
    const size_t n = wm.FieldSize(), c = wm.CodeSize();
    std::vector<float> field(n), afield(c);
    Codes out;
    out.z.resize(data.size() * c);
    out.za.resize(data.size() * c);
    out.next.resize(data.size() * c);
    for (size_t i = 0; i < data.size(); ++i)
    {
        PaintStripes(data[i].obs, field);
        wm.Encode(field, std::span<float>(out.z.data() + i * c, c));
        PaintStripes(data[i].next, field);
        wm.Encode(field, std::span<float>(out.next.data() + i * c, c));
        PaintStripes(data[i].act, afield);
        wm.EncodeAction(afield, std::span<float>(out.za.data() + i * c, c));
    }
    return out;
}

// Mean squared error of Predict against the true next code, over the
// power of the true next code. Identity is the same ratio with z itself
// as the guess: the number to beat.
void Score(WorldModel& wm, const Codes& s, float& model, float& identity)
{
    const size_t c = wm.CodeSize();
    const size_t count = s.z.size() / c;
    std::vector<float> hat(c);
    double err = 0, id = 0, power = 0;
    for (size_t i = 0; i < count; ++i)
    {
        std::span<const float> z(s.z.data() + i * c, c), za(s.za.data() + i * c, c),
            next(s.next.data() + i * c, c);
        wm.Predict(z, za, hat);
        for (size_t j = 0; j < c; ++j)
        {
            err += (hat[j] - next[j]) * (hat[j] - next[j]);
            id += (z[j] - next[j]) * (z[j] - next[j]);
            power += next[j] * next[j];
        }
    }
    model = static_cast<float>(err / power);
    identity = static_cast<float>(id / power);
}

} // namespace

int main()
{
    WorldModelConfig cfg;
    cfg.encoder.dim = kDim;
    cfg.encoder.leak_rate = 0.25f;
    cfg.encoder.input_scaling = 0.8f;
    cfg.encoder.passes = 2 * kDim;
    cfg.k = kK;
    cfg.predictor.z_max = 3 * kK;
    cfg.predictor.gather_span = 5;
    cfg.predictor.tanh_last = true;
    cfg.predictor.training.lr = 0.03f;
    cfg.predictor.training.lr_min_frac = 0.05f;
    cfg.predictor.training.restore_best = true;
    auto wm = WorldModel::Create(cfg);
    const size_t c = wm->CodeSize();

    std::mt19937 rng(0);
    const Codes train = EncodeAll(*wm, Draw(kTrain, rng));
    const Codes val = EncodeAll(*wm, Draw(kVal, rng));
    const Codes test = EncodeAll(*wm, Draw(kTest, rng));

    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        wm->SetEpoch(epoch, kEpochs);
        float loss_sum = 0.f;
        for (int start = 0; start < kTrain; start += kBatch)
        {
            wm->BeginBatch();
            for (int i = start; i < std::min(start + kBatch, kTrain); ++i)
                loss_sum += wm->Accumulate(
                    std::span<const float>(train.z.data() + i * c, c),
                    std::span<const float>(train.za.data() + i * c, c),
                    std::span<const float>(train.next.data() + i * c, c));
            wm->EndBatch();
        }
        float v, v_id;
        Score(*wm, val, v, v_id);
        wm->Observe(v, epoch);
        if (epoch % 50 == 0 || epoch == kEpochs - 1)
            std::printf("epoch=%d mean_loss=%.4f val mse/power=%.4f\n",
                        epoch, loss_sum / kTrain, v);
    }
    wm->RestoreBest();

    float t, t_id;
    Score(*wm, test, t, t_id);
    std::printf("test mse/power: model %.4f  identity %.4f\n", t, t_id);

    // Rollout: H action codes in, H + 1 view codes out. The first H
    // action codes of the test set stand in for a plan.
    const size_t h = 3;
    std::vector<float> path((h + 1) * c);
    wm->Rollout(std::span<const float>(test.z.data(), c),
                std::span<const float>(test.za.data(), h * c), path);

    // Save, Load, and check the reloaded model encodes and predicts the
    // same on a fresh transition.
    wm->Save("quick_start.wm");
    auto again = WorldModel::Load("quick_start.wm");
    const Transition fresh = Draw(1, rng)[0];
    std::vector<float> field(wm->FieldSize()), picture(c);
    std::vector<float> z1(c), za1(c), z2(c), za2(c);
    PaintStripes(fresh.obs, field);
    wm->Encode(field, z1);
    again->Encode(field, z2);
    PaintStripes(fresh.act, picture);
    wm->EncodeAction(picture, za1);
    again->EncodeAction(picture, za2);
    std::vector<float> p1(c), p2(c);
    wm->Predict(z1, za1, p1);
    again->Predict(z2, za2, p2);
    bool same = true;
    for (size_t j = 0; j < c; ++j)
        same = same && z1[j] == z2[j] && za1[j] == za2[j] && p1[j] == p2[j];
    std::printf("reload encodes and predicts the same: %s\n", same ? "yes" : "no");
    return same ? 0 : 1;
}
