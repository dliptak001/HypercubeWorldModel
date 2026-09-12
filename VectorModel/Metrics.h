// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

class VectorModel;

/// MSE of predicting z_{t+1} = z_t. @p z and @p zn are the same length.
[[nodiscard]] float NoChangeMse(std::span<const float> z, std::span<const float> zn);

/// mse / NoChangeMse. NaN if the baseline is 0.
[[nodiscard]] float OneStepRatio(float mse, std::span<const float> z,
                                 std::span<const float> zn);

struct ActionSensitivity
{
    float div = 0.f;   // mean squared difference of the two predictions
    float mse = 0.f;   // one-step val MSE passed in
    float act = 0.f;   // div / mse; NaN if mse is 0
};

/// Same z, two already-run predictions (same length as each other).
[[nodiscard]] ActionSensitivity ActionSensitivityFromPreds(std::span<const float> pred1,
                                                           std::span<const float> pred2,
                                                           float mse);

/// Draw two random action batches, EncodeAction, Predict, and score.
/// Samples in ActionLow/ActionHigh when bounds are set, else [-1, 1].
/// Uses @p n rows or all of @p z, whichever is smaller.
[[nodiscard]] ActionSensitivity MeasureActionSensitivity(VectorModel& vm,
                                                         std::span<const float> z,
                                                         size_t count, float mse,
                                                         uint64_t seed,
                                                         size_t n = 4000);

struct LinearR2
{
    float min = 0.f;
    float mean = 0.f;
    std::vector<float> r2;   // per target dimension
};

/// Per-dimension val R² of a linear ridge from z to y. @p z / @p y are
/// train; @p z_val / @p y_val are val. y is already normalised (or any
/// vector readout). Ridge is on (Z | 1), not multiplied by count.
[[nodiscard]] LinearR2 LinearR2On(std::span<const float> z, std::span<const float> y,
                                  std::span<const float> z_val, std::span<const float> y_val,
                                  size_t train_count, size_t val_count,
                                  size_t z_dim, size_t y_dim, float ridge = 1e-4f);

struct RolloutError
{
    std::vector<float> error;      // H
    std::vector<float> baseline;   // H, no-change from z0
    std::vector<float> ratio;      // error / baseline
};

/// Open-loop latent error at each horizon vs no-change. @p z_pred and
/// @p z_true are (windows × (H+1) × code) row-major; row 0 is z0.
[[nodiscard]] RolloutError RolloutErrorFromCodes(std::span<const float> z_pred,
                                                 std::span<const float> z_true,
                                                 size_t windows, size_t h1,
                                                 size_t code);
