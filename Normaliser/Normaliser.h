// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstddef>
#include <span>
#include <vector>

/// @brief Per-dimension affine map plus clip into [-1, 1].
///
/// Fitted once on a batch of vectors: mean and standard deviation per
/// column, then Apply is (x − mean) / std, clipped at @c clip standard
/// deviations, scaled to [-1, 1]. Clip is a fit parameter, default 3.
/// Copyable. Not a WorldModel member: attach it to a VectorModel, or
/// apply it yourself before PaintStripes.
///
/// Std is floored at 1e-6 so a constant column does not explode.
class Normaliser
{
public:
    /// @brief Fit on @p x, a (count × dim) row-major block.
    /// @throws std::invalid_argument if @p dim is 0, @p x is not a
    ///         multiple of @p dim, count is 0, or @p clip is not finite
    ///         and > 0 (bit-level, holds under fast-math).
    static Normaliser Fit(std::span<const float> x, size_t dim, float clip = 3.f);

    /// @brief Rebuild from saved mean / std / clip.
    /// @throws std::invalid_argument if lengths differ, dim is 0, or
    ///         clip is not finite and > 0.
    static Normaliser FromState(std::span<const float> mean,
                                std::span<const float> std,
                                float clip);

    Normaliser() = default;

    /// @brief Map @p x into [-1, 1]. @p x is one vector (Dim()) or many
    /// concatenated; @p dst is the same length.
    /// @throws std::invalid_argument if lengths are not a multiple of
    ///         Dim(), or @p dst is the wrong length.
    void Apply(std::span<const float> x, std::span<float> dst) const;

    [[nodiscard]] size_t Dim() const { return mean_.size(); }
    [[nodiscard]] float Clip() const { return clip_; }
    [[nodiscard]] const std::vector<float>& Mean() const { return mean_; }
    [[nodiscard]] const std::vector<float>& Std() const { return std_; }

private:
    Normaliser(std::vector<float> mean, std::vector<float> std, float clip);

    std::vector<float> mean_;
    std::vector<float> std_;
    float clip_ = 3.f;
};
