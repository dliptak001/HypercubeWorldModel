// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstddef>
#include <span>
#include <vector>

/// @brief Ridge map from a code (optionally plus an action code) to one
/// number: cost, reward, value, distance — whatever y the host supplies.
///
/// kind is the feature map: linear (the code) or quadratic (the code plus
/// its pairwise products, including squares). sign tells PlanCost whether
/// to negate, so a planner always minimises. za is optional extra features
/// when the target depends on the action; the reacher path omits it.
///
/// Feature count for quadratic is c + c(c+1)/2. Fine through code_size
/// 128 (8,256 features, or twice that with za). Above that is a later
/// fallback, not this class.
///
/// Copyable. Not a WorldModel member: named Heads live on VectorModel.
class Head
{
public:
    enum class Kind { Linear, Quadratic };
    enum class Sign { Cost, Reward };

    struct Score
    {
        float r2 = 0.f;
        float auc = 0.f;          // NaN when not computed
        int positives = 0;
        bool has_auc = false;
    };

    /// @throws std::invalid_argument if @p ridge is not finite and > 0.
    explicit Head(Kind kind = Kind::Quadratic, float ridge = 1e-3f,
                  Sign sign = Sign::Cost);

    /// @brief Fit on @p count rows of codes @p z (count × code, row-major)
    /// against scalar @p y (count). @p za is empty (omitted) or the same
    /// layout as @p z.
    /// @throws std::invalid_argument on empty data, length mismatch, a
    ///         non-finite ridge already checked at construction, or a
    ///         singular solve.
    void Fit(std::span<const float> z, size_t code, std::span<const float> y,
             size_t count, std::span<const float> za = {});

    /// @brief Predict. @p z is one code or many concatenated; @p dst has
    /// one value per row. @p za must be present iff Fit saw za.
    void Apply(std::span<const float> z, std::span<float> dst,
               std::span<const float> za = {}) const;

    /// R² against @p y. AUC when y is strictly 0/1 and both classes are
    /// present (ties count 0.5).
    [[nodiscard]] Score ScoreOn(std::span<const float> z, std::span<const float> y,
                                std::span<const float> za = {}) const;

    /// @brief Planner cost: predicted y summed over the rollout excluding
    /// z0, negated for a reward head. @p zs is (batch × (H+1) × code)
    /// row-major; @p out is batch long.
    /// @throws std::invalid_argument if this Head was fit with za (view
    ///         codes only), or if lengths do not match.
    void PlanCost(std::span<const float> zs, size_t batch, size_t h1,
                  std::span<float> out) const;

    [[nodiscard]] Kind GetKind() const { return kind_; }
    [[nodiscard]] Sign GetSign() const { return sign_; }
    [[nodiscard]] float Ridge() const { return ridge_; }
    [[nodiscard]] bool UsesZa() const { return uses_za_; }
    [[nodiscard]] bool Fitted() const { return !w_.empty(); }
    [[nodiscard]] size_t CodeSize() const { return code_; }
    [[nodiscard]] size_t FeatureCount() const { return w_.size(); }
    [[nodiscard]] float Bias() const { return b_; }
    [[nodiscard]] const std::vector<float>& Weights() const { return w_; }
    [[nodiscard]] const std::vector<float>& Mu() const { return mu_; }
    [[nodiscard]] const std::vector<float>& Sd() const { return sd_; }

    /// Rebuild a fitted Head. @p w, @p mu, @p sd must be the same length.
    static Head FromState(Kind kind, Sign sign, float ridge, bool uses_za,
                          size_t code, std::span<const float> w, float b,
                          std::span<const float> mu, std::span<const float> sd);

private:
    void CheckZa(bool has_za) const;
    void Features(std::span<const float> z, std::span<const float> za,
                  size_t count, std::vector<float>& out) const;
    [[nodiscard]] size_t CodeFeatures() const;

    Kind kind_ = Kind::Quadratic;
    Sign sign_ = Sign::Cost;
    float ridge_ = 1e-3f;
    bool uses_za_ = false;
    size_t code_ = 0;
    std::vector<float> w_, mu_, sd_;
    float b_ = 0.f;
};
