// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "LCN.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/// @brief Trained map from a code (optionally plus an action code) to one
/// number: cost, reward, value, distance — whatever y the host supplies.
///
/// The map is an LCN on the code's cube. Predict is output vertex 0.
/// Fit trains that vertex against y (LCN prefix loss of length 1).
/// sign tells PlanCost whether to negate, so a planner always
/// minimises. za is optional: when present it is packed on the extra
/// bit-face (same concat as WorldModel Pack); the reacher path omits it.
///
/// code_size must be a power of two, at least 16 (LCN dim 4). With za
/// the packed cube is 2 × code_size.
///
/// Copyable (clones LCN weights). Not a WorldModel member: named Heads
/// live on VectorModel.
class Head
{
public:
    enum class Sign { Cost, Reward };

    struct Config
    {
        Sign     sign = Sign::Cost;
        uint64_t seed = 1;
        size_t   z_max = 0;          // LCN depth; 0 → dim of the packed cube (full reach)
        size_t   gather_span = 2;    // [2, 6]
        bool     tanh_last = false;  // false: last depth raw, so y is unscaled
        float    lr = 1e-2f;
        float    lr_min_frac = 0.02f;
        int      lr_decay_epochs = 0;
        bool     restore_best = true;
    };

    struct Score
    {
        float r2 = 0.f;
        float auc = 0.f;          // NaN when not computed
        int positives = 0;
        bool has_auc = false;
    };

    Head();
    explicit Head(Config cfg);
    explicit Head(Sign sign);

    Head(const Head&);
    Head& operator=(const Head&);
    Head(Head&&) noexcept;
    Head& operator=(Head&&) noexcept;
    ~Head();

    /// @brief Fit on codes @p z (count × code_size, row-major) against
    /// scalar @p y (count). @p za is empty (omitted) or the same layout
    /// as @p z. count is y.size().
    /// @throws std::invalid_argument on empty data, length mismatch, or a
    ///         code size the LCN will not take.
    void Fit(std::span<const float> z, size_t code_size, std::span<const float> y,
             int epochs = 40, size_t batch = 32, std::span<const float> za = {});

    /// @brief Predict a scalar per code. @p z is one code or many
    /// concatenated; @p dst has one value per row. @p za must be present
    /// iff Fit saw za.
    void Predict(std::span<const float> z, std::span<float> dst,
                 std::span<const float> za = {}) const;

    /// R² against @p y. AUC when y is strictly 0/1 and both classes are
    /// present (ties count 0.5).
    [[nodiscard]] Score ScoreOn(std::span<const float> z, std::span<const float> y,
                                std::span<const float> za = {}) const;

    /// @brief Planner cost: predicted y summed over the rollout excluding
    /// z0, negated for a reward head. @p zs is (batch × (H+1) × code)
    /// row-major; @p out is batch long. batch and H+1 are inferred.
    /// @throws std::invalid_argument if this Head was fit with za (view
    ///         codes only), or if lengths do not match.
    void PlanCost(std::span<const float> zs, std::span<float> out) const;

    [[nodiscard]] Sign GetSign() const { return cfg_.sign; }
    [[nodiscard]] const Config& GetConfig() const { return cfg_; }
    [[nodiscard]] bool UsesZa() const { return uses_za_; }
    [[nodiscard]] bool Fitted() const { return fitted_; }
    [[nodiscard]] size_t CodeSize() const { return code_; }
    [[nodiscard]] const LCN& Net() const;

    /// Rebuild a fitted Head from saved LCN weights.
    static Head FromState(const Config& cfg, bool uses_za, size_t code_size,
                          std::span<const float> weights);

private:
    void CheckZa(bool has_za) const;
    void EnsureNet(size_t field_n);
    void Pack(const float* z, const float* za, std::span<float> field) const;
    [[nodiscard]] float Readout() const;

    Config cfg_{};
    std::unique_ptr<LCN> net_;
    bool uses_za_ = false;
    bool fitted_ = false;
    size_t code_ = 0;
};
