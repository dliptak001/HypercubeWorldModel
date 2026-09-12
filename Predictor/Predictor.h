// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "LCN.h"
#include "LCNTraining.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/// @brief Construction parameters for @ref Predictor. All fixed at Create.
///
/// LCN fields with the same meaning and ranges as @ref LCNConfig.
/// dim is the latent cube: typically the Encoder k-face, N = 2^dim.
struct PredictorConfig
{
    /// Latent cube dimension; N = 2^dim. Must be in [4, 24].
    size_t dim = 4;

    /// LCN depth; 0 = use dim, else must be >= 2.
    size_t z_max = 0;

    /// LCN lookback window width in fields. Must be in [2, 6].
    size_t gather_span = 2;

    /// If true, the last depth applies tanh.
    bool tanh_last = false;

    /// Seed for the LCN's initial weight draw.
    uint64_t seed = 934791766227647176;

    /// Adam and schedule settings for the LCNTraining the Predictor owns.
    LCNTrainingConfig training;
};

/// @brief Forward map on the k-face: E(x_t) → predicted E(x_{t+1}).
///
/// Owns an LCN on a cube of dimension dim (the latent size, not the
/// Encoder's cube) and an LCNTraining. The caller passes the k-face
/// from Encoder; this class does not include Encoder. Action is not an
/// input here: WorldModel concatenates E(x) and E(a) onto a (k+1)-cube
/// and passes that cube in. The Decoder is not part of this map.
///
/// ```
///   pred.Predict(z, hat);   // z and hat: N = 2^dim floats
/// ```
///
/// Training, one batch:
/// ```
///   pred.BeginBatch();
///   for each pair: pred.Accumulate(z_t, z_next);
///   pred.EndBatch();
/// ```
///
/// There is no Save / Load here: WorldModel persists the Predictor
/// weights. Replicas share them with LoadWeights.
///
/// Non-copyable and non-movable; obtain instances via Create.
/// One instance is not thread-safe for concurrent public calls.
class Predictor
{
public:
    /// @brief Validate @p cfg and build the LCN and LCNTraining.
    /// @throws std::invalid_argument if LCN or LCNTraining reject the fields.
    static std::unique_ptr<Predictor> Create(const PredictorConfig& cfg);

    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;
    Predictor(Predictor&&) = delete;
    Predictor& operator=(Predictor&&) = delete;

    /// @brief One forward step. @p z is E(x_t), @p dst is the predicted
    /// next face; both length Size(). Returns dst.data().
    /// @throws std::invalid_argument if @p z or @p dst is not Size() long.
    const float* Predict(std::span<const float> z, std::span<float> dst);

    /// @brief Clear the accumulated gradient. Call at the start of a batch.
    void BeginBatch();

    /// @brief One training pair: Forward on @p z, Loss against @p next,
    /// Backward into the gradient.
    /// @p next may be shorter than Size(): only the prefix is the target
    /// (LCNTraining prefix loss). The rest of the cube is unconstrained.
    /// @return The sample loss, 0.5 * SSE over the targeted prefix.
    /// @throws std::invalid_argument if @p z is not Size() long, or if
    ///         @p next is empty or longer than Size().
    float Accumulate(std::span<const float> z, std::span<const float> next);

    /// @brief One Adam step on the accumulated gradient.
    void EndBatch();

    /// @brief Apply the cosine schedule for this epoch. See LCNTraining::SetEpoch.
    void SetEpoch(int epoch, int num_epochs = 0);

    /// @brief Report a validation metric; a new low snapshots the weights
    /// when restore_best is on. See LCNTraining::Observe.
    void Observe(float metric, int epoch);

    /// @brief Restore the best snapshot, if any. See LCNTraining::RestoreBest.
    void RestoreBest();

    /// @brief Forget Adam state, learning rate, and best snapshot. Weights
    /// are kept.
    void ResetTraining();

    /// Trainable weights, LCN::Weights() layout and length.
    [[nodiscard]] const std::vector<float>& Weights() const;
    /// Replace the weights. @p w must be Weights().size() long.
    void LoadWeights(std::span<const float> w);
    /// Accumulated gradient, same length as Weights().
    [[nodiscard]] const std::vector<float>& Grad() const;
    /// Add a replica shard into the accumulated gradient.
    void AddGrad(std::span<const float> g);

    /// N = 2^dim. Length of the k-face in and out.
    [[nodiscard]] size_t Size() const { return n_; }

    /// The resolved config (z_max = 0 already replaced by dim).
    [[nodiscard]] const PredictorConfig& Config() const { return cfg_; }

private:
    explicit Predictor(const PredictorConfig& cfg);

    PredictorConfig cfg_;
    size_t n_ = 0;
    std::unique_ptr<LCN> net_;              // must outlive training_
    std::unique_ptr<LCNTraining> training_;
};
