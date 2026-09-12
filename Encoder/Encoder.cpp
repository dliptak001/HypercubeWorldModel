#include "Encoder.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

float MeanAbs(std::span<const float> x)
{
    if (x.empty())
        return 0.f;
    double a = 0.0;
    for (const float v : x)
        a += std::fabs(static_cast<double>(v));
    return static_cast<float>(a / static_cast<double>(x.size()));
}

float Rms(std::span<const float> x)
{
    if (x.empty())
        return 0.f;
    double ss = 0.0;
    for (const float v : x)
        ss += static_cast<double>(v) * static_cast<double>(v);
    return static_cast<float>(std::sqrt(ss / static_cast<double>(x.size())));
}

float SuggestedOutputScale(std::span<const float> raw, float target_rms)
{
    if (raw.empty())
        throw std::invalid_argument("SuggestedOutputScale raw must not be empty");
    if (!FiniteBits(target_rms) || !(target_rms > 0.f))
        throw std::invalid_argument("SuggestedOutputScale target_rms must be finite and > 0");
    for (const float v : raw)
    {
        if (!FiniteBits(v))
            throw std::invalid_argument("SuggestedOutputScale raw contains a non-finite value");
    }
    const float rms = Rms(raw);
    if (!(rms > 0.f))
        throw std::invalid_argument("SuggestedOutputScale raw is all zero");
    return target_rms / rms;
}

// Named substreams (values are part of the weight-draw ABI vs hESN roles).
enum class SeedRole : uint64_t {
    Recurrent = 1,
    Input = 2,
    // 3 reserved (was ExternalFeedback in hESN) — never reuse for a new role
    // that should not collide with historical draws if code is compared.
    // 4 reserved (was Bias) — same rule.
    SrProbe = 5
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Encoder::Encoder(const EncoderConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      spectral_radius_(cfg.spectral_radius),
      leak_rate_(cfg.leak_rate),
      input_scaling_(cfg.input_scaling),
      output_scale_(cfg.output_scale),
      history_depth_(cfg.history_depth),
      passes_(cfg.passes),
      ic_seed_(cfg.ic_seed)
{
    if (dim_ < 5 || dim_ > 24)
        throw std::invalid_argument("Encoder::Create dim must be in [5, 24]");

    n_ = 1ULL << dim_;
    num_input_weights_ = n_ * dim_;

    // Bit-level finiteness test: std::isfinite is unreliable under
    // -ffast-math, and a NaN passes every ordered comparison below.
    if (!FiniteBits(spectral_radius_) || !(spectral_radius_ > 0.0f))
        throw std::invalid_argument("Encoder::Create spectral_radius must be finite and positive");
    if (!FiniteBits(leak_rate_) || !(leak_rate_ > 0.0f) || !(leak_rate_ <= 1.0f))
        throw std::invalid_argument("Encoder::Create leak_rate must be finite and in (0.0, 1.0]");
    if (!FiniteBits(input_scaling_))
        throw std::invalid_argument("Encoder::Create input_scaling must be finite");
    if (!FiniteBits(output_scale_) || !(output_scale_ > 0.0f))
        throw std::invalid_argument("Encoder::Create output_scale must be finite and > 0");
    if (history_depth_ < 1 || history_depth_ > 64)
        throw std::invalid_argument("Encoder::Create history_depth must be in [1, 64]");

    // Weight layout: [ input: N·DIM | recurrent: N·M·DIM ]
    num_weights_ = n_ * dim_ * (history_depth_ + 1u);

    vtx_input_.reset(AllocAligned(n_));
    vtx_state_.reset(AllocAligned(n_));
    vtx_output_history_.reset(AllocAligned(n_ * history_depth_));
    vtx_weight_.reset(AllocAligned(num_weights_));
    output_scaled_.reset(AllocAligned(n_));
    slice_ptrs_.reset(new float*[history_depth_]());

    if (passes_ == 0)
        passes_ = n_;

    Initialize();

    // Episode start state s0: one full delay line, drawn once from ic_seed.
    // Same draw as HypercubeWTF so the same ic_seed gives the same s0.
    s0_.assign(n_ * history_depth_, 0.0f);
    {
        std::mt19937_64 ic_rng(mix64(ic_seed_ ^ 0x5343000000000001ULL));
        std::uniform_real_distribution<float> ic_dist(-0.5f, 0.5f);
        for (float& v : s0_)
            v = ic_dist(ic_rng);
    }
    drive_.assign(n_, 0.0f);
    episode_field_.assign(n_, 0.0f);
}

// ---------------------------------------------------------------------------
// Weight draw + spectral-radius rescale
// ---------------------------------------------------------------------------

void Encoder::Initialize()
{
    auto seed_for = [this](SeedRole r) {
        return mix64(rng_seed_ ^ (0x100000001B3ULL * static_cast<uint64_t>(r)));
    };
    std::mt19937_64 rng(seed_for(SeedRole::Recurrent));
    std::mt19937_64 in_rng(seed_for(SeedRole::Input));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Clear();

    float* pW = vtx_weight_.get();

    float* const input_base = pW;
    for (size_t i = 0; i < num_input_weights_; ++i)
        (*pW++) = static_cast<float>(dist(in_rng));
    const float in_scaling = input_scaling_ / std::sqrt(static_cast<float>(dim_));
    for (size_t i = 0; i < num_input_weights_; ++i)
        input_base[i] *= in_scaling;

    const size_t rec_base = RecurrentWeightBase();
    const float w_scaling =
        1.0f / std::sqrt(static_cast<float>(dim_ * history_depth_));
    for (size_t i = rec_base; i < num_weights_; ++i)
        vtx_weight_[i] = static_cast<float>(dist(rng)) * w_scaling;

    const float target = spectral_radius_;
    const size_t MN = history_depth_ * n_;
    std::vector<float> sr_x(MN, 0.0f), sr_y(MN, 0.0f);
    {
        std::mt19937_64 sr_rng(seed_for(SeedRole::SrProbe));
        std::uniform_real_distribution<double> sr_dist(-1.0, 1.0);
        float norm = 0.0f;
        for (size_t v = 0; v < n_; ++v)
        {
            sr_x[v] = static_cast<float>(sr_dist(sr_rng));
            norm += sr_x[v] * sr_x[v];
        }
        norm = std::sqrt(norm);
        for (size_t v = 0; v < n_; ++v)
            sr_x[v] /= norm;
    }

    float applied_scale = 1.0f;
    auto eval_sr = [&](float s) {
        const float rel = s / applied_scale;
        for (size_t i = rec_base; i < num_weights_; ++i)
            vtx_weight_[i] *= rel;
        applied_scale = s;
        return EstimateSpectralRadius(sr_x, sr_y);
    };

    const float pre_sr = EstimateSpectralRadius(sr_x, sr_y);
    float post_sr = pre_sr;
    int sr_iters = 0;
    if (pre_sr > 1e-6f)
    {
        constexpr float kSrTolRel = 0.001f;
        constexpr int kMaxSrIters = 20;

        float s0 = 1.0f, h0 = pre_sr - target;
        float s1 = target / pre_sr, h1 = eval_sr(s1) - target;
        ++sr_iters;
        post_sr = h1 + target;
        while (sr_iters < kMaxSrIters &&
               std::abs(post_sr - target) > target * kSrTolRel)
        {
            const float denom = h1 - h0;
            float s2 = (std::abs(denom) < 1e-12f)
                           ? s1 * (target / std::max(post_sr, 1e-6f))
                           : s1 - h1 * (s1 - s0) / denom;
            s2 = std::clamp(s2, 0.25f * s1, 4.0f * s1);
            post_sr = eval_sr(s2);
            ++sr_iters;
            s0 = s1;
            h0 = h1;
            s1 = s2;
            h1 = post_sr - target;
        }
    }
    realized_spectral_radius_ = post_sr;
}

// ---------------------------------------------------------------------------
// Episode
// ---------------------------------------------------------------------------

const float* Encoder::RunEpisode(std::span<const float> x)
{
    if (x.size() != n_)
        throw std::invalid_argument("Encoder::RunEpisode: x.size() must equal N = 2^dim");

    // Snapshot before LoadInitialCondition overwrites the delay line, so a
    // span into RawCube (or ScaledCube) is a valid field.
    if (x.data() != episode_field_.data())
        std::memcpy(episode_field_.data(), x.data(), n_ * sizeof(float));

    LoadInitialCondition(s0_.data(), s0_.size());

    float* drive = drive_.data();
    const float* src = episode_field_.data();
    const size_t n_mask = n_ - 1;
    size_t c = 0;
    for (size_t pass = 0; pass < passes_; ++pass)
    {
        for (size_t v = 0; v < n_; ++v)
            drive[v] = src[(v ^ c) & n_mask];
        InjectInputField(drive, n_);
        Step();
        ++c;
    }
    has_episode_ = true;
    RefreshScaledOutput();
    return output_scaled_.get();
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

void Encoder::Step()
{
    const float* p_vtx_prev = slice_ptrs_[0];
    for (size_t v = 0; v < n_; v++)
        UpdateState(v, p_vtx_prev[v]);

    float* p0 = slice_ptrs_[history_depth_ - 1];
    for (size_t i = history_depth_ - 1; i > 0; --i)
        slice_ptrs_[i] = slice_ptrs_[i - 1];
    slice_ptrs_[0] = p0;

    std::memcpy(slice_ptrs_[0], vtx_state_.get(), n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
}

void Encoder::UpdateState(const size_t v, const float old_output_v)
{
    float s = 0.0f;
    const float* iw = vtx_weight_.get() + v * dim_;
    const float* w =
        &vtx_weight_[RecurrentWeightBase()] + v * dim_ * history_depth_;

    for (size_t i = 0; i < dim_; i++)
        s += vtx_input_[v ^ NearestMask(i)] * iw[i];

    for (size_t i = 0; i < history_depth_; i++)
    {
        const float* pSlice = slice_ptrs_[i];
        for (size_t j = 0; j < dim_; j++)
            s += pSlice[v ^ NearestMask(j)] * (*w++);
    }

    const float activation = std::tanh(s);
    vtx_state_[v] = (1.0f - leak_rate_) * old_output_v + leak_rate_ * activation;
}

// ---------------------------------------------------------------------------
// Drive injection / IC
// ---------------------------------------------------------------------------

void Encoder::InjectInputField(const float* field, const size_t count)
{
    if (field == nullptr)
        throw std::invalid_argument("Encoder::InjectInputField: field is null");
    if (count != n_)
        throw std::invalid_argument(
            "Encoder::InjectInputField: count must equal N = 2^dim");
    std::memcpy(vtx_input_.get(), field, n_ * sizeof(float));
}

void Encoder::HomeSlicePointers()
{
    for (size_t i = 0; i < history_depth_; i++)
        slice_ptrs_[i] = &vtx_output_history_[i * n_];
}

void Encoder::LoadInitialCondition(const float* ic, const size_t count)
{
    if (ic == nullptr)
        throw std::invalid_argument("Encoder::LoadInitialCondition: ic is null");
    const size_t need = n_ * history_depth_;
    if (count != need)
        throw std::invalid_argument(
            "Encoder::LoadInitialCondition: count must equal N * history_depth");

    // Canonical ring home, then bulk load logical ages 0..M-1 into physical slots.
    HomeSlicePointers();
    std::memcpy(vtx_output_history_.get(), ic, need * sizeof(float));
    std::memcpy(vtx_state_.get(), ic, n_ * sizeof(float)); // age-0
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
}

// ---------------------------------------------------------------------------
// Config / clear
// ---------------------------------------------------------------------------

const float* Encoder::RawCube() const
{
    if (!has_episode_)
        throw std::invalid_argument("Encoder::RawCube requires a prior RunEpisode");
    return slice_ptrs_[0];
}

const float* Encoder::ScaledCube() const
{
    if (!has_episode_)
        throw std::invalid_argument("Encoder::ScaledCube requires a prior RunEpisode");
    return output_scaled_.get();
}

float Encoder::SuggestOutputScale(float target_rms) const
{
    if (!has_episode_)
        throw std::invalid_argument("Encoder::SuggestOutputScale requires a prior RunEpisode");
    return SuggestedOutputScale(std::span<const float>(slice_ptrs_[0], n_), target_rms);
}

float Encoder::SuggestOutputScale(std::span<const float> fields, float target_rms)
{
    if (fields.empty() || fields.size() % n_ != 0)
        throw std::invalid_argument(
            "Encoder::SuggestOutputScale fields length must be a positive multiple of N");
    const size_t count = fields.size() / n_;
    std::vector<float> raw(fields.size());
    for (size_t i = 0; i < count; ++i)
    {
        RunEpisode(fields.subspan(i * n_, n_));
        std::memcpy(raw.data() + i * n_, slice_ptrs_[0], n_ * sizeof(float));
    }
    return SuggestedOutputScale(raw, target_rms);
}

void Encoder::FitOutputScale(float target_rms)
{
    SetOutputScale(SuggestOutputScale(target_rms));
}

void Encoder::FitOutputScale(std::span<const float> fields, float target_rms)
{
    SetOutputScale(SuggestOutputScale(fields, target_rms));
}

void Encoder::SetOutputScale(float scale)
{
    if (!FiniteBits(scale) || !(scale > 0.f))
        throw std::invalid_argument("Encoder::SetOutputScale scale must be finite and > 0");
    output_scale_ = scale;
    if (has_episode_)
        RefreshScaledOutput();
}

void Encoder::RefreshScaledOutput()
{
    const float* raw = slice_ptrs_[0];
    float* dst = output_scaled_.get();
    const float s = output_scale_;
    for (size_t i = 0; i < n_; ++i)
        dst[i] = raw[i] * s;
}

EncoderConfig Encoder::Config() const
{
    EncoderConfig cfg;
    cfg.dim = dim_;
    cfg.seed = rng_seed_;
    cfg.spectral_radius = spectral_radius_;
    cfg.leak_rate = leak_rate_;
    cfg.input_scaling = input_scaling_;
    cfg.history_depth = history_depth_;
    cfg.passes = passes_;
    cfg.ic_seed = ic_seed_;
    cfg.output_scale = output_scale_;
    return cfg;
}

void Encoder::Clear()
{
    std::memset(vtx_state_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_output_history_.get(), 0, n_ * history_depth_ * sizeof(float));
    HomeSlicePointers();
}

// ---------------------------------------------------------------------------
// Spectral radius (companion operator on MN-dimensional delay state)
// ---------------------------------------------------------------------------

float Encoder::EstimateSpectralRadius(std::span<float> x, std::span<float> y) const
{
    const size_t MN = history_depth_ * n_;
    if (x.size() < MN || y.size() < MN)
        throw std::invalid_argument(
            "Encoder::EstimateSpectralRadius: probe buffers must hold M * N floats");

    constexpr int kMaxIters = 1500;
    constexpr int kBurnIn = 32;
    constexpr int kCheckSpacing = 50;
    constexpr float kTolRel = 1e-4f;

    float rho_ring[kCheckSpacing] = {};
    double sum_log = 0.0;
    int n_acc = 0;
    float rho = 0.0f;

    for (int iter = 0; iter < kMaxIters; ++iter)
    {
        for (size_t v = 0; v < n_; v++)
        {
            float s = 0.0f;
            const float* w =
                &vtx_weight_[RecurrentWeightBase()] + v * dim_ * history_depth_;
            for (size_t j = 0; j < history_depth_; j++)
            {
                const float* x_j = x.data() + j * n_;
                const float* wj = w + j * dim_;
                for (size_t i = 0; i < dim_; i++)
                    s += wj[i] * x_j[v ^ NearestMask(i)];
            }
            y[v] = s;
        }

        for (size_t j = 1; j < history_depth_; j++)
            std::memcpy(y.data() + j * n_, x.data() + (j - 1) * n_,
                        n_ * sizeof(float));

        float norm = 0.0f;
        for (size_t k = 0; k < MN; k++)
            norm += y[k] * y[k];
        norm = std::sqrt(norm);
        if (norm <= 1e-30f)
            return 0.0f;

        const float inv = 1.0f / norm;
        for (size_t k = 0; k < MN; k++)
            x[k] = y[k] * inv;

        if (iter < kBurnIn)
            continue;

        sum_log += std::log(static_cast<double>(norm));
        ++n_acc;
        rho = static_cast<float>(std::exp(sum_log / static_cast<double>(n_acc)));

        const int slot = n_acc % kCheckSpacing;
        if (n_acc > kCheckSpacing &&
            std::abs(rho - rho_ring[slot]) < rho * kTolRel)
            break;
        rho_ring[slot] = rho;
    }

    return rho;
}
