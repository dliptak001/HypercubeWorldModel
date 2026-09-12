// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "Encoder.h"
#include "Predictor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

/// @brief Paint a short vector onto a field as contiguous stripes.
///
/// Cell j of @p dst takes src[j * src.size() / dst.size()], so value i
/// of @p src fills a block of about dst.size() / src.size() cells and
/// every cell of @p dst is written. This is how a state vector or an
/// action vector that is much shorter than the cube becomes a field the
/// encoder can work with: one narrow value becomes a wide stripe. A
/// caller with its own picture writes the field directly instead.
/// @throws std::invalid_argument if @p src is empty or longer than @p dst.
void PaintStripes(std::span<const float> src, std::span<float> dst);

/// @brief Predictor knobs inside @ref WorldModelConfig. dim is not here:
/// it is always k+1.
struct WorldModelPredictorConfig
{
    /// Predictor LCN depth; 0 = use k+1, else must be >= 2.
    size_t z_max = 0;

    /// Predictor LCN lookback window width in fields. Must be in [2, 6].
    size_t gather_span = 3;

    /// If true, the Predictor's last depth applies tanh.
    bool tanh_last = false;

    /// Seed for the Predictor's LCN weight draw. Encoder has its own seed.
    uint64_t seed = 934791766227647176;

    /// Adam and schedule settings for the Predictor.
    LCNTrainingConfig training;
};

/// @brief Construction parameters for @ref WorldModel. All fixed at Create.
///
/// Encoder knobs are @ref EncoderConfig. The action encoder is not a
/// separate knob: it takes the same EncoderConfig with dim replaced by
/// k, so its cube is the k-face size and its full output is E(a).
/// Predictor dim is not a knob either: it is always k+1 (first subcube
/// holds E(x); extra bit-face holds E(a)). k must be in [5, encoder.dim).
struct WorldModelConfig
{
    EncoderConfig encoder{};

    /// Code face dimension. Compression is this cut. Strictly less than
    /// encoder.dim; the action encoder needs k >= 5. Predictor cube is k+1.
    size_t k = 6;

    WorldModelPredictorConfig predictor{};
};

/// @brief Frozen perception plus trained latent dynamics.
///
/// Owns two Encoders and a Predictor. Encode runs a view episode on the
/// dim-cube and writes the low k-face (the compression) into a buffer
/// the caller owns. EncodeAction runs an action episode on a k-cube and
/// writes its whole output, also 2^k. Predict maps E(x) cat E(a) to a
/// predicted next E(x). Neither Encoder trains. The Decoder is not part
/// of this object.
///
/// The Predictor cube is k+1. Pack concatenates two length-2^k codes:
/// E(x) on the first subcube, E(a) on the extra bit-face. Loudness is
/// each Encoder's output_scale, applied when Encode / EncodeAction
/// return, not at Pack. Return and loss are the first subcube only
/// (length 2^k). How the caller paints the action field is not this
/// class's job.
///
/// ```
///   wm.Encode(x, z);                      // x: FieldSize(); z: CodeSize()
///   wm.EncodeAction(a_field, za);         // a_field and za: CodeSize()
///   const float* hat = wm.Predict(z, za); // hat: CodeSize()
/// ```
/// Encode does not keep a code. Two windows need two buffers or the
/// second Encode overwrites the first. Predict's pointer is valid until
/// the next Predict, Rollout, or Accumulate. Rollout chains Predict over
/// a sequence of action codes into caller storage.
///
/// A view or action given as a short vector rather than a field goes
/// through @ref PaintStripes first. Save and Load carry the config and
/// the Predictor weights; the encoders are rebuilt from their seeds.
///
/// Training is on k-faces, same cycle as Predictor. Encode the stream
/// once into caller storage, EncodeAction each action field, then
/// Accumulate:
/// ```
///   wm.BeginBatch();
///   for each pair: wm.Accumulate(z_t, za_t, z_next);
///   wm.EndBatch();
/// ```
///
/// Non-copyable and non-movable; obtain instances via Create or Load.
/// One instance is not thread-safe for concurrent public calls, except
/// Pack: const and no member buffer, so threads may Pack concurrently
/// into distinct dst buffers.
class WorldModel
{
public:
    /// Library version; matches the CMake project VERSION.
    static constexpr const char kVersion[] = "1.1.0";

    /// @brief Validate @p cfg, build both Encoders and the Predictor.
    /// @throws std::invalid_argument if k is not in [5, encoder.dim),
    ///         or if Encoder / Predictor reject their fields.
    static std::unique_ptr<WorldModel> Create(const WorldModelConfig& cfg);

    /// @brief Read a WorldModel written by @ref Save.
    /// @throws std::runtime_error if the file cannot be opened, the magic
    ///         or version is wrong, the weight count does not match the
    ///         config in the file, or the file is truncated.
    /// @throws std::invalid_argument if the config in the file is invalid.
    static std::unique_ptr<WorldModel> Load(const std::filesystem::path& file);

    /// @brief Write the config and the Predictor weights to @p file. The
    /// encoders are not written: they are frozen and rebuilt from their
    /// seeds on Load. encoder.passes is written as it was given to
    /// Create, 0 included, so Load resolves it per cube exactly as
    /// Create did (view T = N, action T = 2^k). Adam state is not
    /// written.
    /// @throws std::runtime_error if the file cannot be written.
    void Save(const std::filesystem::path& file) const;

    /// Same payload as @ref Save to a path, written to an already-open
    /// binary stream. Used by VectorModel to embed a WorldModel.
    void Save(std::ostream& os) const;

    /// @brief Read a WorldModel from an already-open binary stream.
    /// @p source is the name used in error messages.
    static std::unique_ptr<WorldModel> Load(std::istream& is, std::string_view source);

    WorldModel(const WorldModel&) = delete;
    WorldModel& operator=(const WorldModel&) = delete;
    WorldModel(WorldModel&&) = delete;
    WorldModel& operator=(WorldModel&&) = delete;

    /// @brief Frozen encode. @p field is one window, length FieldSize().
    /// Writes the k-face into @p dst (length CodeSize()) and returns
    /// dst.data(). The pointer lives as long as @p dst does.
    /// @throws std::invalid_argument if @p field is not FieldSize() long
    ///         or @p dst is not CodeSize() long.
    const float* Encode(std::span<const float> field, std::span<float> dst);

    /// Scaled full view episode from the most recent Encode. Length
    /// FieldSize(). Valid until the next Encode. The k-face Encode
    /// wrote is the first CodeSize() of this cube.
    /// @throws std::invalid_argument if Encode has not been called.
    [[nodiscard]] const float* LastCube() const;

    /// Unscaled full view episode from the most recent Encode. Length
    /// FieldSize(). Valid until the next Encode. The raw k-face is the
    /// first CodeSize() of this cube.
    /// @throws std::invalid_argument if Encode has not been called.
    [[nodiscard]] const float* LastRawCube() const;

    /// Packed Predictor input from the most recent Predict or
    /// Accumulate: E(x) then E(a), length 2 * CodeSize(). The two Pack
    /// halves are [0, CodeSize) and [CodeSize, 2 * CodeSize).
    /// @throws std::invalid_argument if neither Predict nor Accumulate
    ///         has been called.
    [[nodiscard]] const float* LastPacked() const;

    /// @brief Frozen action encode on the k-cube. @p field is one action
    /// field, length CodeSize(). Writes the whole k-cube output, E(a),
    /// into @p dst (length CodeSize()) and returns dst.data(). Nothing
    /// is cut on this side: the action cube is already the k-face size.
    /// @throws std::invalid_argument if @p field or @p dst is not
    ///         CodeSize() long.
    const float* EncodeAction(std::span<const float> field, std::span<float> dst);

    /// @brief Predict the next k-face. @p z is E(x), @p a is E(a);
    /// both length CodeSize().
    /// @return Predicted next k-face (first subcube), until the next
    ///         Predict, Rollout, or Accumulate. Rollout calls Predict,
    ///         so after a Rollout the pointer holds its last step.
    /// @throws std::invalid_argument if @p z or @p a is not CodeSize() long.
    const float* Predict(std::span<const float> z, std::span<const float> a);

    /// @brief Chain Predict over a sequence of action codes.
    /// @p z0 is E(x) at the start, length CodeSize(). @p actions is H
    /// action codes E(a) laid end to end, length H * CodeSize(), each
    /// already through EncodeAction. Writes H + 1 codes into @p out,
    /// length (H + 1) * CodeSize(): out[0] is z0, out[t + 1] is the
    /// prediction from out[t] and actions[t]. H may be 0.
    /// The Predict pointer is invalidated.
    /// @throws std::invalid_argument if @p z0 is not CodeSize() long,
    ///         @p actions is not a multiple of CodeSize(), or @p out is
    ///         not (H + 1) * CodeSize() long.
    void Rollout(std::span<const float> z0, std::span<const float> actions,
                 std::span<float> out);

    /// @brief Concatenate two k-faces: E(x) then E(a).
    /// @p dst is length 2 * CodeSize(). Const and touches no member
    /// buffer, so threads may call it concurrently, each into its own
    /// dst; replica Predictors feed on it (see WorldModelTest).
    /// @throws std::invalid_argument if @p z, @p a, or @p dst is the
    ///         wrong length.
    void Pack(std::span<const float> z, std::span<const float> a,
              std::span<float> dst) const;

    /// @brief Clear the accumulated gradient. Call at the start of a batch.
    void BeginBatch();

    /// @brief One training pair: pack E(x) cat E(a), Forward, Loss on
    /// the first subcube against @p next, Backward. Extra bit-face
    /// unconstrained.
    /// @return The sample loss, 0.5 * SSE over the k-face.
    /// @throws std::invalid_argument if @p z, @p a, or @p next is not
    ///         CodeSize() long.
    float Accumulate(std::span<const float> z, std::span<const float> a,
                     std::span<const float> next);

    /// @brief One Adam step on the accumulated gradient.
    void EndBatch();

    /// @brief Apply the cosine schedule for this epoch. See LCNTraining::SetEpoch.
    void SetEpoch(int epoch, int num_epochs = 0);

    /// @brief Report a validation metric; a new low snapshots the weights
    /// when restore_best is on. See LCNTraining::Observe.
    void Observe(float metric, int epoch);

    /// @brief Restore the best snapshot, if any. See LCNTraining::RestoreBest.
    void RestoreBest();

    /// @brief Forget Adam state, learning rate, and best snapshot. Encoder
    /// and Predictor weights are kept.
    void ResetTraining();

    /// Predictor weights, same layout and length as Predictor::Weights().
    [[nodiscard]] const std::vector<float>& Weights() const;
    /// Replace the Predictor weights. @p w must be Weights().size() long.
    void LoadWeights(std::span<const float> w);
    /// Accumulated Predictor gradient, same length as Weights().
    [[nodiscard]] const std::vector<float>& Grad() const;
    /// Add a replica shard into the accumulated gradient.
    void AddGrad(std::span<const float> g);

    /// Encoder cube length: N = 2^encoder.dim. Length of an Encode field.
    [[nodiscard]] size_t FieldSize() const;

    /// k-face length: 2^k. Length of Encode output, of an action field
    /// and its E(a), and of Predict in/out. The Predictor cube is twice
    /// this (dim = k+1).
    [[nodiscard]] size_t CodeSize() const;

    [[nodiscard]] size_t K() const { return cfg_.k; }

    /// The resolved config (encoder.passes and predictor.z_max already filled in).
    /// encoder.passes here is the view encoder's T. Do not feed this
    /// snapshot back to Create: a passes given as 0 resolves to N here
    /// but to 2^k on the action encoder, so a rebuild from Config()
    /// would give the action encoder N passes. Use Save / Load, or keep
    /// the config you built.
    [[nodiscard]] const WorldModelConfig& Config() const { return cfg_; }

    /// encoder.passes as it was given to Create, 0 included. This is what
    /// Save writes, and what a host that serializes its own config should
    /// keep: Config().encoder.passes is the view encoder's resolved T and
    /// would give the action encoder the wrong T on a rebuild.
    [[nodiscard]] size_t RequestedPasses() const { return requested_passes_; }

    /// The action encoder's resolved config: cfg.encoder with dim = k.
    [[nodiscard]] EncoderConfig ActionEncoderConfig() const;

    [[nodiscard]] float ViewOutputScale() const;
    [[nodiscard]] float ActionOutputScale() const;
    void SetViewOutputScale(float scale);
    void SetActionOutputScale(float scale);

    /// @brief Scale that would bring already-run **raw** view codes to
    /// RMS @p target_rms. Does not change the Encoder. @p z is one
    /// k-face or many concatenated.
    [[nodiscard]] float SuggestViewOutputScale(std::span<const float> z,
                                               float target_rms = 1.f) const;
    [[nodiscard]] float SuggestActionOutputScale(std::span<const float> za,
                                                 float target_rms = 1.f) const;
    void FitViewOutputScale(std::span<const float> z, float target_rms = 1.f);
    void FitActionOutputScale(std::span<const float> za, float target_rms = 1.f);

    [[nodiscard]] float RealizedSpectralRadius() const;
    [[nodiscard]] float ActionRealizedSpectralRadius() const;

private:
    explicit WorldModel(const WorldModelConfig& cfg);

    WorldModelConfig cfg_;
    size_t requested_passes_ = 0;   // encoder.passes as given to Create; what Save writes
    std::unique_ptr<Encoder> enc_;
    std::unique_ptr<Encoder> act_enc_;
    std::unique_ptr<Predictor> pred_;
    std::vector<float> packed_;
    const float* last_cube_ = nullptr;
    bool has_packed_ = false;
};
