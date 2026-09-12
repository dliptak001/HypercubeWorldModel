// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// VectorModel sibling of quick_start.cpp. A point on a plane moves by a
// bounded velocity. The VectorModel paints, encodes, and rolls raw
// actions; a Head scores squared distance to a goal.

#include "VectorModel.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <span>
#include <vector>

namespace {

constexpr size_t kDim = 6, kK = 5;
constexpr float kStep = 0.1f;
constexpr int kTrain = 256, kEpochs = 80, kBatch = 16;

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

    auto vm = VectorModel::Create(cfg);
    const float lo[2] = {-1.f, -1.f}, hi[2] = {1.f, 1.f};
    vm->SetActionBounds(lo, hi);
    const size_t c = vm->CodeSize();

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> obs(kTrain * 2), act(kTrain * 2), nxt(kTrain * 2);
    std::vector<float> z(kTrain * c), za(kTrain * c), zn(kTrain * c);
    std::vector<float> y(kTrain);
    const float goal[2] = {0.6f, -0.4f};
    for (int i = 0; i < kTrain; ++i)
    {
        obs[i * 2] = u(rng);
        obs[i * 2 + 1] = u(rng);
        act[i * 2] = u(rng);
        act[i * 2 + 1] = u(rng);
        nxt[i * 2] = std::clamp(obs[i * 2] + kStep * act[i * 2], -1.f, 1.f);
        nxt[i * 2 + 1] = std::clamp(obs[i * 2 + 1] + kStep * act[i * 2 + 1], -1.f, 1.f);
        const float dx = nxt[i * 2] - goal[0], dy = nxt[i * 2 + 1] - goal[1];
        y[i] = dx * dx + dy * dy;
        vm->Encode(std::span<const float>(obs.data() + i * 2, 2),
                   std::span<float>(z.data() + i * c, c));
        vm->EncodeAction(std::span<const float>(act.data() + i * 2, 2),
                         std::span<float>(za.data() + i * c, c));
        vm->Encode(std::span<const float>(nxt.data() + i * 2, 2),
                   std::span<float>(zn.data() + i * c, c));
    }

    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        vm->SetEpoch(epoch, kEpochs);
        for (int start = 0; start < kTrain; start += kBatch)
        {
            vm->BeginBatch();
            for (int i = start; i < std::min(start + kBatch, kTrain); ++i)
                vm->Accumulate(std::span<const float>(z.data() + i * c, c),
                               std::span<const float>(za.data() + i * c, c),
                               std::span<const float>(zn.data() + i * c, c));
            vm->EndBatch();
        }
        std::vector<float> hat(c);
        double err = 0, power = 0;
        for (int i = 0; i < kTrain; ++i)
        {
            vm->Predict(std::span<const float>(z.data() + i * c, c),
                        std::span<const float>(za.data() + i * c, c), hat);
            for (size_t j = 0; j < c; ++j)
            {
                const float d = hat[j] - zn[i * c + j];
                err += d * d;
                power += zn[i * c + j] * zn[i * c + j];
            }
        }
        vm->Observe(static_cast<float>(err / power), epoch);
    }
    vm->RestoreBest();

    Head head;
    head.Fit(zn, c, y, 40, 16);
    const Head::Score sc = head.ScoreOn(zn, y);
    std::printf("Head R2 on toy distance %.4f\n", sc.r2);
    vm->SetHead("dist2", head);

    std::vector<float> z0(c), path((3 + 1) * c);
    vm->Encode(std::span<const float>(obs.data(), 2), z0);
    vm->Rollout(z0, std::span<const float>(act.data(), 3 * 2), path);
    std::vector<float> cost(1);
    vm->Cost(path, cost);
    std::printf("raw-action rollout H=3 cost %.4f\n", cost[0]);

    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "vector_start.hvm";
    vm->Save(file);
    auto again = VectorModel::Load(file);
    std::filesystem::remove(file);
    std::vector<float> z2(c);
    again->Encode(std::span<const float>(obs.data(), 2), z2);
    bool same = true;
    for (size_t j = 0; j < c; ++j)
        same = same && z0[j] == z2[j];
    std::printf("reload encodes the same: %s\n", same ? "yes" : "no");
    return (sc.r2 > 0.9f && same) ? 0 : 1;
}
