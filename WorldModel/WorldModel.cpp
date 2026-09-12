// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "WorldModel.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr char kMagic[4] = {'H', 'W', 'M', '1'};
constexpr uint32_t kFileVersionV1 = 1;
constexpr uint32_t kFileVersion = 2;

template <typename T>
void WriteRaw(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T ReadRaw(std::istream& is, const char* what)
{
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!is)
        throw std::runtime_error(std::string("WorldModel::Load truncated reading ") + what);
    return v;
}

} // namespace

void PaintStripes(std::span<const float> src, std::span<float> dst)
{
    if (src.empty())
        throw std::invalid_argument("PaintStripes src must not be empty");
    if (src.size() > dst.size())
        throw std::invalid_argument("PaintStripes src must not be longer than dst");
    const size_t n = dst.size();
    const size_t m = src.size();
    for (size_t j = 0; j < n; ++j)
        dst[j] = src[j * m / n];
}

std::unique_ptr<WorldModel> WorldModel::Create(const WorldModelConfig& cfg)
{
    return std::unique_ptr<WorldModel>(new WorldModel(cfg));
}

WorldModel::WorldModel(const WorldModelConfig& cfg)
    : cfg_(cfg), requested_passes_(cfg.encoder.passes)
{
    enc_ = Encoder::Create(cfg.encoder);
    const size_t dim = enc_->Config().dim;
    if (cfg.k < 5 || cfg.k >= dim)
        throw std::invalid_argument(
            "WorldModel::Create k must be in [5, encoder.dim)");

    // action encoder: same knobs and seeds, cube of dimension k; its
    // whole output is E(a), already the k-face size. output_scale starts
    // the same as the view encoder; SetActionOutputScale after Create.
    EncoderConfig acfg = cfg.encoder;
    acfg.dim = cfg.k;
    act_enc_ = Encoder::Create(acfg);

    PredictorConfig pcfg;
    pcfg.dim = cfg.k + 1;
    pcfg.z_max = cfg.predictor.z_max;
    pcfg.gather_span = cfg.predictor.gather_span;
    pcfg.tanh_last = cfg.predictor.tanh_last;
    pcfg.seed = cfg.predictor.seed;
    pcfg.training = cfg.predictor.training;
    pred_ = Predictor::Create(pcfg);

    cfg_.encoder = enc_->Config();
    cfg_.predictor.z_max = pred_->Config().z_max;
    packed_.resize(pred_->Size());
}

const float* WorldModel::Encode(std::span<const float> field, std::span<float> dst)
{
    if (dst.size() != CodeSize())
        throw std::invalid_argument(
            "WorldModel::Encode dst must be CodeSize() long");
    last_cube_ = enc_->RunEpisode(field);
    const size_t sub = dst.size();
    for (size_t i = 0; i < sub; ++i)
        dst[i] = last_cube_[i];
    return dst.data();
}

const float* WorldModel::LastCube() const
{
    if (last_cube_ == nullptr)
        throw std::invalid_argument("WorldModel::LastCube requires a prior Encode");
    return last_cube_;
}

const float* WorldModel::LastRawCube() const
{
    if (last_cube_ == nullptr)
        throw std::invalid_argument("WorldModel::LastRawCube requires a prior Encode");
    return enc_->RawCube();
}

const float* WorldModel::LastPacked() const
{
    if (!has_packed_)
        throw std::invalid_argument(
            "WorldModel::LastPacked requires a prior Predict or Accumulate");
    return packed_.data();
}

const float* WorldModel::EncodeAction(std::span<const float> field, std::span<float> dst)
{
    const size_t sub = CodeSize();
    if (field.size() != sub)
        throw std::invalid_argument(
            "WorldModel::EncodeAction field must be CodeSize() long");
    if (dst.size() != sub)
        throw std::invalid_argument(
            "WorldModel::EncodeAction dst must be CodeSize() long");
    const float* cube = act_enc_->RunEpisode(field);
    for (size_t i = 0; i < sub; ++i)
        dst[i] = cube[i];
    return dst.data();
}

void WorldModel::Pack(std::span<const float> z, std::span<const float> a,
                      std::span<float> dst) const
{
    const size_t sub = CodeSize();
    if (z.size() != sub)
        throw std::invalid_argument(
            "WorldModel::Pack z must be CodeSize() long");
    if (a.size() != sub)
        throw std::invalid_argument(
            "WorldModel::Pack a must be CodeSize() long");
    if (dst.size() != pred_->Size())
        throw std::invalid_argument(
            "WorldModel::Pack dst must be 2 * CodeSize() long");
    for (size_t i = 0; i < sub; ++i)
        dst[i] = z[i];
    for (size_t i = 0; i < sub; ++i)
        dst[sub + i] = a[i];
}

const float* WorldModel::Predict(std::span<const float> z, std::span<const float> a)
{
    Pack(z, a, packed_);
    has_packed_ = true;
    return pred_->Predict(packed_);
}

void WorldModel::Rollout(std::span<const float> z0, std::span<const float> actions,
                         std::span<float> out)
{
    const size_t sub = CodeSize();
    if (z0.size() != sub)
        throw std::invalid_argument(
            "WorldModel::Rollout z0 must be CodeSize() long");
    if (actions.size() % sub != 0)
        throw std::invalid_argument(
            "WorldModel::Rollout actions must be a multiple of CodeSize() long");
    const size_t h = actions.size() / sub;
    if (out.size() != (h + 1) * sub)
        throw std::invalid_argument(
            "WorldModel::Rollout out must be (H + 1) * CodeSize() long");
    for (size_t i = 0; i < sub; ++i)
        out[i] = z0[i];
    for (size_t t = 0; t < h; ++t)
    {
        const std::span<const float> z(out.data() + t * sub, sub);
        const std::span<const float> a(actions.data() + t * sub, sub);
        const float* hat = Predict(z, a);
        float* next = out.data() + (t + 1) * sub;
        for (size_t i = 0; i < sub; ++i)
            next[i] = hat[i];
    }
}

void WorldModel::Save(const std::filesystem::path& file) const
{
    std::ofstream os(file, std::ios::binary);
    if (!os)
        throw std::runtime_error("WorldModel::Save cannot open " + file.string());
    Save(os);
    if (!os)
        throw std::runtime_error("WorldModel::Save write failed for " + file.string());
}

void WorldModel::Save(std::ostream& os) const
{
    os.write(kMagic, sizeof(kMagic));
    WriteRaw(os, kFileVersion);

    const EncoderConfig& e = cfg_.encoder;
    WriteRaw(os, static_cast<uint64_t>(e.dim));
    WriteRaw(os, static_cast<uint64_t>(e.seed));
    WriteRaw(os, e.spectral_radius);
    WriteRaw(os, e.leak_rate);
    WriteRaw(os, e.input_scaling);
    WriteRaw(os, static_cast<uint64_t>(e.history_depth));
    WriteRaw(os, static_cast<uint64_t>(requested_passes_));   // not the resolved view T
    WriteRaw(os, static_cast<uint64_t>(e.ic_seed));

    WriteRaw(os, static_cast<uint64_t>(cfg_.k));
    WriteRaw(os, enc_->OutputScale());
    WriteRaw(os, act_enc_->OutputScale());
    WriteRaw(os, static_cast<uint64_t>(cfg_.predictor.z_max));
    WriteRaw(os, static_cast<uint64_t>(cfg_.predictor.gather_span));
    WriteRaw(os, static_cast<uint8_t>(cfg_.predictor.tanh_last ? 1 : 0));
    WriteRaw(os, static_cast<uint64_t>(cfg_.predictor.seed));

    const LCNTrainingConfig& t = cfg_.predictor.training;
    WriteRaw(os, t.lr);
    WriteRaw(os, t.lr_min_frac);
    WriteRaw(os, static_cast<int32_t>(t.lr_decay_epochs));
    WriteRaw(os, static_cast<uint8_t>(t.restore_best ? 1 : 0));
    WriteRaw(os, t.beta1);
    WriteRaw(os, t.beta2);
    WriteRaw(os, t.eps);

    const std::vector<float>& w = pred_->Weights();
    WriteRaw(os, static_cast<uint64_t>(w.size()));
    os.write(reinterpret_cast<const char*>(w.data()),
             static_cast<std::streamsize>(w.size() * sizeof(float)));
}

std::unique_ptr<WorldModel> WorldModel::Load(const std::filesystem::path& file)
{
    std::ifstream is(file, std::ios::binary);
    if (!is)
        throw std::runtime_error("WorldModel::Load cannot open " + file.string());
    return Load(is, file.string());
}

std::unique_ptr<WorldModel> WorldModel::Load(std::istream& is, std::string_view source)
{
    const std::string src(source);
    char magic[4];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        throw std::runtime_error("WorldModel::Load bad magic in " + src);

    const uint32_t version = ReadRaw<uint32_t>(is, "version");
    if (version != kFileVersion && version != kFileVersionV1)
        throw std::runtime_error("WorldModel::Load unsupported version " +
                                 std::to_string(version) + " in " + src);

    WorldModelConfig cfg;
    cfg.encoder.dim = static_cast<size_t>(ReadRaw<uint64_t>(is, "encoder.dim"));
    cfg.encoder.seed = ReadRaw<uint64_t>(is, "encoder.seed");
    cfg.encoder.spectral_radius = ReadRaw<float>(is, "encoder.spectral_radius");
    cfg.encoder.leak_rate = ReadRaw<float>(is, "encoder.leak_rate");
    cfg.encoder.input_scaling = ReadRaw<float>(is, "encoder.input_scaling");
    cfg.encoder.history_depth = static_cast<size_t>(ReadRaw<uint64_t>(is, "encoder.history_depth"));
    cfg.encoder.passes = static_cast<size_t>(ReadRaw<uint64_t>(is, "encoder.passes"));
    cfg.encoder.ic_seed = ReadRaw<uint64_t>(is, "encoder.ic_seed");

    cfg.k = static_cast<size_t>(ReadRaw<uint64_t>(is, "k"));
    float action_output_scale = 1.f;
    if (version == kFileVersionV1)
    {
        // v1 Pack multiplied E(a) by action_scale; view codes were raw.
        action_output_scale = ReadRaw<float>(is, "action_scale");
        cfg.encoder.output_scale = 1.f;
    }
    else
    {
        cfg.encoder.output_scale = ReadRaw<float>(is, "view_output_scale");
        action_output_scale = ReadRaw<float>(is, "action_output_scale");
    }
    cfg.predictor.z_max = static_cast<size_t>(ReadRaw<uint64_t>(is, "z_max"));
    cfg.predictor.gather_span = static_cast<size_t>(ReadRaw<uint64_t>(is, "gather_span"));
    cfg.predictor.tanh_last = ReadRaw<uint8_t>(is, "tanh_last") != 0;
    cfg.predictor.seed = ReadRaw<uint64_t>(is, "seed");

    cfg.predictor.training.lr = ReadRaw<float>(is, "lr");
    cfg.predictor.training.lr_min_frac = ReadRaw<float>(is, "lr_min_frac");
    cfg.predictor.training.lr_decay_epochs = ReadRaw<int32_t>(is, "lr_decay_epochs");
    cfg.predictor.training.restore_best = ReadRaw<uint8_t>(is, "restore_best") != 0;
    cfg.predictor.training.beta1 = ReadRaw<float>(is, "beta1");
    cfg.predictor.training.beta2 = ReadRaw<float>(is, "beta2");
    cfg.predictor.training.eps = ReadRaw<float>(is, "eps");

    const uint64_t n_weights = ReadRaw<uint64_t>(is, "n_weights");

    auto wm = Create(cfg);   // invalid_argument here means a bad config in the file
    wm->SetActionOutputScale(action_output_scale);
    if (n_weights != wm->pred_->Weights().size())
        throw std::runtime_error("WorldModel::Load weight count " + std::to_string(n_weights) +
                                 " does not match config in " + src);

    std::vector<float> w(static_cast<size_t>(n_weights));
    is.read(reinterpret_cast<char*>(w.data()),
            static_cast<std::streamsize>(w.size() * sizeof(float)));
    if (!is)
        throw std::runtime_error("WorldModel::Load truncated reading weights in " + src);

    wm->pred_->LoadWeights(w);
    return wm;
}

void WorldModel::BeginBatch()
{
    pred_->BeginBatch();
}

float WorldModel::Accumulate(std::span<const float> z, std::span<const float> a,
                             std::span<const float> next)
{
    if (next.size() != CodeSize())
        throw std::invalid_argument(
            "WorldModel::Accumulate next must be CodeSize() long");
    Pack(z, a, packed_);
    has_packed_ = true;
    return pred_->Accumulate(packed_, next);
}

void WorldModel::EndBatch()
{
    pred_->EndBatch();
}

void WorldModel::SetEpoch(int epoch, int num_epochs)
{
    pred_->SetEpoch(epoch, num_epochs);
}

void WorldModel::Observe(float metric, int epoch)
{
    pred_->Observe(metric, epoch);
}

void WorldModel::RestoreBest()
{
    pred_->RestoreBest();
}

void WorldModel::ResetTraining()
{
    pred_->ResetTraining();
}

const std::vector<float>& WorldModel::Weights() const
{
    return pred_->Weights();
}

void WorldModel::LoadWeights(std::span<const float> w)
{
    pred_->LoadWeights(w);
}

const std::vector<float>& WorldModel::Grad() const
{
    return pred_->Grad();
}

void WorldModel::AddGrad(std::span<const float> g)
{
    pred_->AddGrad(g);
}

size_t WorldModel::FieldSize() const
{
    return enc_->Size();
}

size_t WorldModel::CodeSize() const
{
    return size_t{1} << cfg_.k;
}

EncoderConfig WorldModel::ActionEncoderConfig() const
{
    return act_enc_->Config();
}

float WorldModel::ViewOutputScale() const
{
    return enc_->OutputScale();
}

float WorldModel::ActionOutputScale() const
{
    return act_enc_->OutputScale();
}

void WorldModel::SetViewOutputScale(float scale)
{
    enc_->SetOutputScale(scale);
    cfg_.encoder.output_scale = enc_->OutputScale();
}

void WorldModel::SetActionOutputScale(float scale)
{
    act_enc_->SetOutputScale(scale);
}

float WorldModel::SuggestViewOutputScale(std::span<const float> z, float target_rms) const
{
    return SuggestedOutputScale(z, target_rms);
}

float WorldModel::SuggestActionOutputScale(std::span<const float> za, float target_rms) const
{
    return SuggestedOutputScale(za, target_rms);
}

void WorldModel::FitViewOutputScale(std::span<const float> z, float target_rms)
{
    SetViewOutputScale(SuggestViewOutputScale(z, target_rms));
}

void WorldModel::FitActionOutputScale(std::span<const float> za, float target_rms)
{
    SetActionOutputScale(SuggestActionOutputScale(za, target_rms));
}

float WorldModel::RealizedSpectralRadius() const
{
    return enc_->RealizedSpectralRadius();
}

float WorldModel::ActionRealizedSpectralRadius() const
{
    return act_enc_->RealizedSpectralRadius();
}
