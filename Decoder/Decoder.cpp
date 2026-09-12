// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Decoder.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

static_assert(std::endian::native == std::endian::little,
              "Decoder file format is little-endian; big-endian hosts are not supported");

// Bit-level finiteness test: std::isfinite can be folded to true under
// -ffast-math, this cannot.
static bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

std::unique_ptr<Decoder> Decoder::Create(const DecoderConfig& cfg)
{
    return std::unique_ptr<Decoder>(new Decoder(cfg));
}

Decoder::Decoder(const DecoderConfig& cfg)
    : cfg_(cfg)
{
    if (cfg.k >= cfg.dim)
        throw std::invalid_argument(
            "Decoder::Create k must be strictly less than dim");

    net_ = LCN::Create(LCNConfig{.dim = cfg.dim,
                                 .seed = cfg.seed,
                                 .z_max = cfg.z_max,
                                 .gather_span = cfg.gather_span,
                                 .tanh_last = cfg.tanh_last});
    training_ = std::make_unique<LCNTraining>(*net_, cfg.training);

    cfg_.z_max = net_->ZMax();
    n_ = net_->N();
    code_size_ = size_t{1} << cfg.k;
    field_.assign(n_, 0.f);
}

// ---------------------------------------------------------------------------
// Placement, inference, training cycle
// ---------------------------------------------------------------------------

void Decoder::Place(std::span<const float> code)
{
    if (code.size() != code_size_)
        throw std::invalid_argument("Decoder::Place code must be CodeSize() long");
    const float s = input_scale_;
    for (size_t i = 0; i < code_size_; ++i)
        field_[i] = code[i] * s;
    // field_[code_size_ .. N) was zeroed at construction and is never written
}

const float* Decoder::Decode(std::span<const float> code, std::span<float> dst)
{
    if (dst.size() != n_)
        throw std::invalid_argument("Decoder::Decode dst must be FieldSize() long");
    Place(code);
    net_->Forward(field_);
    std::memcpy(dst.data(), net_->Output().data(), n_ * sizeof(float));
    return dst.data();
}

void Decoder::BeginBatch()
{
    training_->ZeroGrad();
}

float Decoder::Accumulate(std::span<const float> code, std::span<const float> target)
{
    if (target.size() != n_)
        throw std::invalid_argument("Decoder::Accumulate target must be FieldSize() long");
    Place(code);
    net_->Forward(field_);
    const float loss = training_->Loss(target);
    training_->Backward();
    return loss;
}

void Decoder::EndBatch()
{
    training_->Adam();
}

void Decoder::SetEpoch(int epoch, int num_epochs)
{
    training_->SetEpoch(epoch, num_epochs);
}

void Decoder::Observe(float metric, int epoch)
{
    training_->Observe(metric, epoch);
}

void Decoder::RestoreBest()
{
    training_->RestoreBest();
}

void Decoder::ResetTraining()
{
    training_->Reset();
}

const std::vector<float>& Decoder::Weights() const
{
    return net_->Weights();
}

void Decoder::LoadWeights(std::span<const float> w)
{
    net_->LoadWeights(w);
}

const std::vector<float>& Decoder::Grad() const
{
    return training_->Grad();
}

void Decoder::AddGrad(std::span<const float> g)
{
    training_->AddGrad(g);
}

// ---------------------------------------------------------------------------
// Input scale
// ---------------------------------------------------------------------------

void Decoder::FitInputScale(std::span<const float> codes)
{
    if (codes.empty())
        throw std::invalid_argument("Decoder::FitInputScale input is empty");
    float peak = 0.f;
    for (const float x : codes)
    {
        if (!FiniteBits(x))
            throw std::invalid_argument("Decoder::FitInputScale input contains a non-finite value");
        peak = std::max(peak, std::abs(x));
    }
    if (peak <= 0.f)
        throw std::invalid_argument("Decoder::FitInputScale input is all zero");
    input_scale_ = 1.f / peak;
}

void Decoder::SetInputScale(float scale)
{
    if (!FiniteBits(scale) || !(scale > 0.f))
        throw std::invalid_argument("Decoder::SetInputScale scale must be finite and > 0");
    input_scale_ = scale;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

namespace
{
constexpr char kMagic[4] = {'H', 'S', 'D', 'C'};
constexpr uint32_t kFileVersion = 1;

template <typename T>
void WriteRaw(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T ReadRaw(std::istream& is, const char* what)
{
    T v;
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!is)
        throw std::runtime_error(std::string("Decoder::Load truncated reading ") + what);
    return v;
}
} // namespace

void Decoder::Save(const std::filesystem::path& file) const
{
    std::ofstream os(file, std::ios::binary);
    if (!os)
        throw std::runtime_error("Decoder::Save cannot open " + file.string());

    os.write(kMagic, sizeof(kMagic));
    WriteRaw(os, kFileVersion);

    WriteRaw(os, static_cast<uint64_t>(cfg_.dim));
    WriteRaw(os, static_cast<uint64_t>(cfg_.k));
    WriteRaw(os, static_cast<uint64_t>(cfg_.z_max));
    WriteRaw(os, static_cast<uint64_t>(cfg_.gather_span));
    WriteRaw(os, static_cast<uint8_t>(cfg_.tanh_last ? 1 : 0));
    WriteRaw(os, static_cast<uint64_t>(cfg_.seed));

    const LCNTrainingConfig& t = cfg_.training;
    WriteRaw(os, t.lr);
    WriteRaw(os, t.lr_min_frac);
    WriteRaw(os, static_cast<int32_t>(t.lr_decay_epochs));
    WriteRaw(os, static_cast<uint8_t>(t.restore_best ? 1 : 0));
    WriteRaw(os, t.beta1);
    WriteRaw(os, t.beta2);
    WriteRaw(os, t.eps);

    WriteRaw(os, input_scale_);

    const std::vector<float>& w = net_->Weights();
    WriteRaw(os, static_cast<uint64_t>(w.size()));
    os.write(reinterpret_cast<const char*>(w.data()),
             static_cast<std::streamsize>(w.size() * sizeof(float)));

    if (!os)
        throw std::runtime_error("Decoder::Save write failed for " + file.string());
}

std::unique_ptr<Decoder> Decoder::Load(const std::filesystem::path& file)
{
    std::ifstream is(file, std::ios::binary);
    if (!is)
        throw std::runtime_error("Decoder::Load cannot open " + file.string());

    char magic[4];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        throw std::runtime_error("Decoder::Load bad magic in " + file.string());

    const uint32_t version = ReadRaw<uint32_t>(is, "version");
    if (version != kFileVersion)
        throw std::runtime_error("Decoder::Load unsupported version " +
                                 std::to_string(version) + " in " + file.string());

    DecoderConfig cfg;
    cfg.dim = static_cast<size_t>(ReadRaw<uint64_t>(is, "dim"));
    cfg.k = static_cast<size_t>(ReadRaw<uint64_t>(is, "k"));
    cfg.z_max = static_cast<size_t>(ReadRaw<uint64_t>(is, "z_max"));
    cfg.gather_span = static_cast<size_t>(ReadRaw<uint64_t>(is, "gather_span"));
    cfg.tanh_last = ReadRaw<uint8_t>(is, "tanh_last") != 0;
    cfg.seed = ReadRaw<uint64_t>(is, "seed");

    cfg.training.lr = ReadRaw<float>(is, "lr");
    cfg.training.lr_min_frac = ReadRaw<float>(is, "lr_min_frac");
    cfg.training.lr_decay_epochs = ReadRaw<int32_t>(is, "lr_decay_epochs");
    cfg.training.restore_best = ReadRaw<uint8_t>(is, "restore_best") != 0;
    cfg.training.beta1 = ReadRaw<float>(is, "beta1");
    cfg.training.beta2 = ReadRaw<float>(is, "beta2");
    cfg.training.eps = ReadRaw<float>(is, "eps");

    const float input_scale = ReadRaw<float>(is, "input_scale");
    const uint64_t n_weights = ReadRaw<uint64_t>(is, "n_weights");

    auto dec = Create(cfg);   // invalid_argument here means a bad config in the file
    if (n_weights != dec->net_->Weights().size())
        throw std::runtime_error("Decoder::Load weight count " + std::to_string(n_weights) +
                                 " does not match config in " + file.string());

    std::vector<float> w(static_cast<size_t>(n_weights));
    is.read(reinterpret_cast<char*>(w.data()),
            static_cast<std::streamsize>(w.size() * sizeof(float)));
    if (!is)
        throw std::runtime_error("Decoder::Load truncated reading weights in " + file.string());

    dec->net_->LoadWeights(w);
    dec->SetInputScale(input_scale);
    return dec;
}
