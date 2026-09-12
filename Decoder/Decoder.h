// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "LCN.h"
#include "LCNTraining.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

/// @brief Construction parameters for @ref Decoder. All fixed at Create.
///
/// The LCN fields (dim, z_max, gather_span, tanh_last, seed) carry the
/// same meaning and ranges as @ref LCNConfig; k is the one field the
/// Decoder adds. It is always strictly less than dim: the encoder is a
/// compression engine, and the Decoder never sees the full encoder cube.
struct DecoderConfig
{
    /// Output cube dimension; the LCN runs on N = 2^dim vertices.
    /// Same dim as the encoder it is paired with. Default matches EncoderConfig.
    size_t dim = 10;

    /// Input face dimension. The Decoder accepts 2^k values and places
    /// them on the low k-face of the LCN. Must be strictly less than dim
    /// — the rest of the encoder cube is the compression, and is not an
    /// input. Default matches WorldModelConfig::k.
    size_t k = 6;

    /// LCN depth; 0 = use dim, else must be >= 2.
    size_t z_max = 0;

    /// LCN lookback window width in fields. Must be in [2, 6].
    size_t gather_span = 2;

    /// If true, the LCN's last depth applies tanh.
    bool tanh_last = false;

    /// Seed for the LCN's initial weight draw.
    uint64_t seed = 934791766227647176;

    /// Adam and schedule settings for the LCNTraining the Decoder owns.
    LCNTrainingConfig training{.restore_best = true};
};

/// @brief Invert a compressed encoder state: code in, field out.
///
/// The encoder holds N = 2^dim values after an episode. Only a k-face
/// of that state (k < dim) is kept; that is the compression. This
/// Decoder takes those 2^k values, scales them by one frozen constant,
/// places them on the LCN's low k-face with every other vertex fed
/// zero, and runs the LCN forward. The output is the reconstructed
/// field, length N. Comparing it to the field that was encoded is how
/// you see what the compression threw away.
///
/// Owns an LCN and an LCNTraining. The Decoder does not include the
/// encoder; the caller slices the encoder output down to CodeSize().
///
/// Inference:
/// ```
///   dec.Decode(code, field);      // field: N floats
/// ```
/// Training, one batch:
/// ```
///   dec.BeginBatch();
///   for each sample: dec.Accumulate(code, target);
///   dec.EndBatch();
/// ```
/// Save and Load carry config, input scale, and weights. Adam state is
/// not saved.
///
/// Non-copyable and non-movable; obtain instances via Create or Load.
/// One instance is not thread-safe for concurrent public calls.
class Decoder
{
public:
    /// @brief Validate @p cfg and build the LCN and LCNTraining.
    /// @throws std::invalid_argument if k is not strictly less than dim,
    ///         or if LCN or LCNTraining reject their own fields.
    static std::unique_ptr<Decoder> Create(const DecoderConfig& cfg);

    /// @brief Read a Decoder written by @ref Save.
    /// @throws std::runtime_error if the file cannot be opened, the magic
    ///         or version is wrong, the weight count does not match the
    ///         config in the file, or the file is truncated.
    /// @throws std::invalid_argument if the config in the file is invalid.
    static std::unique_ptr<Decoder> Load(const std::filesystem::path& file);

    /// @brief Write config, input scale, and weights to @p file.
    /// @throws std::runtime_error if the file cannot be written.
    void Save(const std::filesystem::path& file) const;

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) = delete;
    Decoder& operator=(Decoder&&) = delete;

    /// @brief Inference. Places @p code and runs the LCN forward.
    /// Writes the reconstructed field into @p dst (FieldSize()) and
    /// returns dst.data().
    /// @throws std::invalid_argument if @p code is not CodeSize() long
    ///         or @p dst is not FieldSize() long.
    const float* Decode(std::span<const float> code, std::span<float> dst);

    /// @brief Clear the accumulated gradient. Call at the start of a batch.
    void BeginBatch();

    /// @brief One training sample: place @p code, Forward, Loss against
    /// @p target over the full cube, Backward into the gradient.
    /// @return The sample loss, 0.5 * SSE.
    /// @throws std::invalid_argument if @p code is not CodeSize() long
    ///         or @p target is not FieldSize() long.
    float Accumulate(std::span<const float> code, std::span<const float> target);

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
    /// and input scale are kept.
    void ResetTraining();

    /// Trainable weights, LCN::Weights() layout and length.
    [[nodiscard]] const std::vector<float>& Weights() const;
    /// Replace the weights. @p w must be Weights().size() long.
    void LoadWeights(std::span<const float> w);
    /// Accumulated gradient, same length as Weights().
    [[nodiscard]] const std::vector<float>& Grad() const;
    /// Sum @p g onto the accumulated gradient. Same length as Weights().
    void AddGrad(std::span<const float> g);

    /// @brief Set the input scale to 1 / max |x| over @p codes, which is
    /// the concatenation of the training codes. The scaled input then
    /// lies in [-1, 1].
    /// @throws std::invalid_argument if @p codes is empty, all zero, or
    ///         contains a non-finite value.
    void FitInputScale(std::span<const float> codes);

    /// @brief Set the input scale directly.
    /// @throws std::invalid_argument if @p scale is not finite and positive.
    void SetInputScale(float scale);

    [[nodiscard]] float InputScale() const { return input_scale_; }

    /// Number of input values: 2^k.
    [[nodiscard]] size_t CodeSize() const { return code_size_; }

    /// Number of output values: N = 2^dim. Length of the reconstructed field.
    [[nodiscard]] size_t FieldSize() const { return n_; }

    /// The resolved config (z_max = 0 already replaced by dim).
    [[nodiscard]] const DecoderConfig& Config() const { return cfg_; }

private:
    explicit Decoder(const DecoderConfig& cfg);

    /// Scale @p code and write it to the low k-face of field_.
    /// The tail (vertices outside the input face) stays zero.
    void Place(std::span<const float> code);

    DecoderConfig cfg_;
    size_t n_ = 0;
    size_t code_size_ = 0;
    std::unique_ptr<LCN> net_;              // must outlive training_
    std::unique_ptr<LCNTraining> training_;
    std::vector<float> field_;              // length N; tail stays zero
    float input_scale_ = 1.f;
};
