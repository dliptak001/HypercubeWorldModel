// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "LCN.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/// @brief Trained map from a view code to a raw action vector.
///
/// Same LCN family as Head. Head reads vertex 0 (scalar y). Actor
/// reads vertices 0 .. act_dim-1 (the action). No za, no sign, no
/// plan_cost: those belong to Head. Clip to the action box is the
/// host's job.
///
/// code_size must be a power of two, at least 16 (LCN dim 4).
/// act_dim must be in [1, code_size].
///
/// Copyable (clones LCN weights). One instance is not thread-safe for
/// concurrent public calls: Predict and ScoreOn write the LCN's
/// forward state (logical const).
class Actor
{
public:
    struct Config
    {
        uint64_t seed = 1;
        size_t   z_max = 0;          // LCN depth; 0 → dim of the code cube
        size_t   gather_span = 2;    // [2, 6]
        bool     tanh_last = false;  // false: last depth raw
        float    lr = 1e-2f;
        float    lr_min_frac = 0.02f;
        int      lr_decay_epochs = 0;
        bool     restore_best = true;
    };

    struct Score
    {
        float r2 = 0.f;
    };

    Actor();
    explicit Actor(Config cfg);

    Actor(const Actor&);
    Actor& operator=(const Actor&);
    Actor(Actor&&) noexcept;
    Actor& operator=(Actor&&) noexcept;
    ~Actor();

    /// @brief Fit on codes @p z (count × code_size) against actions
    /// @p a (count × act_dim), both row-major. count is a.size()/act_dim.
    void Fit(std::span<const float> z, size_t code_size,
             std::span<const float> a, size_t act_dim,
             int epochs = 40, size_t batch = 32);

    /// @brief Predict one action per code. @p z is count × code_size;
    /// @p dst is count × act_dim.
    void Predict(std::span<const float> z, std::span<float> dst) const;

    /// R² of flattened actions against @p a.
    [[nodiscard]] Score ScoreOn(std::span<const float> z,
                                std::span<const float> a) const;

    [[nodiscard]] const Config& GetConfig() const { return cfg_; }
    [[nodiscard]] bool Fitted() const { return fitted_; }
    [[nodiscard]] size_t CodeSize() const { return code_; }
    [[nodiscard]] size_t ActDim() const { return act_; }
    [[nodiscard]] const LCN& Net() const;

    static Actor FromState(const Config& cfg, size_t code_size, size_t act_dim,
                           std::span<const float> weights);

private:
    void EnsureNet(size_t field_n);
    void ForwardOne(const float* z, float* dst) const;

    Config cfg_{};
    std::unique_ptr<LCN> net_;
    bool fitted_ = false;
    size_t code_ = 0;
    size_t act_ = 0;
    mutable std::vector<float> field_;
};
