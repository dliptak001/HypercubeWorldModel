// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

/// @brief Construction-time parameters for @ref Encoder.
///
/// All fields except @c output_scale are fixed at @ref Encoder::Create.
/// @c output_scale is the Create default; SetOutputScale / FitOutputScale
/// may change it. Dynamics (state, history, staged drive) are not part
/// of this struct. @ref Config returns an equivalent snapshot;
/// @c spectral_radius is the **target** used for the recurrent rescale,
/// not the realized estimate.
struct EncoderConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[5, 24]**.
    size_t dim = 10;

    /// Master RNG seed for weight draws (named substreams in Encoder.cpp).
    uint64_t seed = 7934791766227647176;

    /// Target spectral radius for the **recurrent** weight block only. Finite, > 0.
    float spectral_radius = 0.999f;

    /// Leaky-integrator mix: 1 = full replacement each step; in (0, 1] blends. Finite.
    float leak_rate = 1.0f;

    /// Input drive strength. Input weights U(-1,1) then × input_scaling / √dim. Finite.
    float input_scaling = 0.02f;

    /// Delay-line length M. Valid range **[1, 64]**.
    size_t history_depth = 8;

    /// Passes per episode (T). 0 means T = N, a full tour of the cube.
    size_t passes = 0;

    /// Seed for the episode start state s0. Separate from @c seed (weights).
    uint64_t ic_seed = 1;

    /// Presentation gain on the cube @ref Encoder::RunEpisode returns.
    /// Finite, > 0. Default 1: the returned slice equals the raw delay
    /// line. Does not enter Step. @ref Encoder::SetOutputScale /
    /// @ref Encoder::FitOutputScale may change it after Create.
    float output_scale = 1.f;
};

/// @brief Mean of absolute values. Empty span is 0.
[[nodiscard]] float MeanAbs(std::span<const float> x);

/// @brief Root-mean-square. Empty span is 0.
[[nodiscard]] float Rms(std::span<const float> x);

/// @brief Scale that would bring @p raw to RMS @p target_rms.
///
/// Returns target_rms / rms(raw). Default target 1. Measures the
/// values as given; pass an unscaled cube or k-face.
/// @throws std::invalid_argument if @p raw is empty or all zero, if
///         it contains a non-finite, or if @p target_rms is not
///         finite and > 0.
[[nodiscard]] float SuggestedOutputScale(std::span<const float> raw,
                                         float target_rms = 1.f);

/// @brief Frozen hypercube encoder: one field in, one episode, N floats out.
///
/// ```
///   const float* out = enc.RunEpisode(x);   // x has N floats; out has N floats
/// ```
/// @ref RunEpisode reloads the start state s0, drives T passes with the
/// field re-addressed by XOR with the pass counter, and returns the
/// newest slice. Nothing carries over between episodes. The pointer is
/// valid until the next RunEpisode. @ref Config is resolved (passes of
/// 0 already replaced by N).
///
/// The returned cube is the full N. A caller who compresses keeps a
/// k-face (the first 2^k) and discards the rest.
///
/// Non-copyable and non-movable; obtain instances only via @ref Create.
/// One instance is not thread-safe for concurrent public calls.
class Encoder
{
public:
    /// @brief Validate @p cfg, allocate, draw weights, rescale recurrent SR.
    /// @throws std::invalid_argument if dim is not in [5, 24],
    ///         spectral_radius is not finite and > 0, leak_rate is not
    ///         finite and in (0, 1], input_scaling is not finite,
    ///         output_scale is not finite and > 0, or history_depth is
    ///         not in [1, 64].
    static std::unique_ptr<Encoder> Create(const EncoderConfig& cfg)
    {
        return std::unique_ptr<Encoder>(new Encoder(cfg));
    }

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) = delete;
    Encoder& operator=(Encoder&&) = delete;

    /// @brief Run one episode on the field @p x.
    ///
    /// Loads s0 into the delay line, sets the pass counter c to 0, then for
    /// T passes drives vertex v with x[(v XOR c) AND (N-1)], steps, and
    /// increments c. @p x is not modified. The returned pointer is the
    /// newest slice times @ref OutputScale, N floats, valid until the
    /// next RunEpisode. The delay line is not scaled.
    /// @throws std::invalid_argument if @p x is not length N.
    const float* RunEpisode(std::span<const float> x);

    /// Unscaled newest delay-line slice from the most recent
    /// @ref RunEpisode. N floats, valid until the next RunEpisode.
    /// @throws std::invalid_argument if RunEpisode has not been called.
    [[nodiscard]] const float* RawCube() const;

    /// Scaled newest slice from the most recent @ref RunEpisode
    /// (the same pointer RunEpisode returned). N floats, valid until
    /// the next RunEpisode.
    /// @throws std::invalid_argument if RunEpisode has not been called.
    [[nodiscard]] const float* ScaledCube() const;

    /// @brief Scale that would bring the last raw cube to RMS @p target_rms.
    /// Does not change @ref OutputScale.
    /// @throws std::invalid_argument if RunEpisode has not been called,
    ///         or if @p target_rms is not finite and > 0.
    [[nodiscard]] float SuggestOutputScale(float target_rms = 1.f) const;

    /// @brief Run each length-N field in @p fields (concatenated) and
    /// return the scale that would bring those raw cubes to RMS
    /// @p target_rms. Does not change @ref OutputScale. Leaves the last
    /// field's episode in the delay line.
    /// @throws std::invalid_argument if @p fields is empty, not a
    ///         multiple of N, or if the raw cubes are all zero.
    [[nodiscard]] float SuggestOutputScale(std::span<const float> fields,
                                           float target_rms = 1.f);

    /// @brief Set @ref OutputScale from @ref SuggestOutputScale on the
    /// last raw cube. Refreshes the scaled buffer from that cube.
    void FitOutputScale(float target_rms = 1.f);

    /// @brief Set @ref OutputScale from a calibration run of @p fields.
    void FitOutputScale(std::span<const float> fields, float target_rms = 1.f);

    /// Presentation gain in effect. Finite, > 0.
    [[nodiscard]] float OutputScale() const { return output_scale_; }

    /// @brief Replace the presentation gain. Finite, > 0. If an episode
    /// has been run, refreshes the scaled buffer from the current raw
    /// cube. Does not rerun the episode.
    /// @throws std::invalid_argument if @p scale is not finite and > 0.
    void SetOutputScale(float scale);

    /// Number of vertices: N = 2^dim. Length of a field and of the
    /// pointer returned by @ref RunEpisode.
    [[nodiscard]] size_t Size() const { return n_; }

    /// The resolved config (passes = 0 already replaced by N).
    [[nodiscard]] EncoderConfig Config() const;

    [[nodiscard]] float RealizedSpectralRadius() const
    {
        return realized_spectral_radius_;
    }

private:
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    explicit Encoder(const EncoderConfig& cfg);

    struct AlignedFree
    {
        void operator()(float* p) const noexcept
        {
            ::operator delete[](p, std::align_val_t{64});
        }
    };

    static float* AllocAligned(size_t count)
    {
        return static_cast<float*>(
            ::operator new[](count * sizeof(float), std::align_val_t{64}));
    }

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;
    size_t n_ = 0;
    size_t num_input_weights_ = 0;

    std::unique_ptr<float[], AlignedFree> vtx_input_;
    std::unique_ptr<float[], AlignedFree> vtx_state_;
    std::unique_ptr<float[], AlignedFree> vtx_output_history_;
    std::unique_ptr<float[], AlignedFree> vtx_weight_;
    std::unique_ptr<float[], AlignedFree> output_scaled_;
    std::unique_ptr<float*[]> slice_ptrs_;

    float spectral_radius_ = 0.99f;
    float leak_rate_ = 1.0f;
    float input_scaling_ = 0.5f;
    float output_scale_ = 1.f;
    float realized_spectral_radius_ = 0.0f;
    bool has_episode_ = false;
    size_t history_depth_ = 1;
    size_t num_weights_ = 0;

    size_t passes_ = 0;               // T
    uint64_t ic_seed_ = 1;
    std::vector<float> s0_;           // N * M, drawn once at construction
    std::vector<float> drive_;        // N, scratch for the re-addressed field

    void Initialize();
    void Step();
    void UpdateState(size_t v, float old_output_v);
    void InjectInputField(const float* field, size_t count);
    void Clear();
    void LoadInitialCondition(const float* ic, size_t count);
    void HomeSlicePointers();
    void RefreshScaledOutput();
    [[nodiscard]] float EstimateSpectralRadius(std::span<float> x, std::span<float> y) const;

    /// Recurrent block starts after the input block.
    [[nodiscard]] size_t RecurrentWeightBase() const { return num_input_weights_; }
};
