// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "VectorModel.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr char kWorldMagic[4] = {'H', 'W', 'M', '1'};

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
        throw std::runtime_error(std::string("VectorModel::Load truncated reading ") + what);
    return v;
}

void WriteBytes(std::ostream& os, const void* p, size_t n)
{
    os.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
}

void WriteString(std::ostream& os, std::string_view s)
{
    WriteRaw(os, static_cast<uint32_t>(s.size()));
    WriteBytes(os, s.data(), s.size());
}

std::string ReadString(std::istream& is, const char* what)
{
    const uint32_t n = ReadRaw<uint32_t>(is, what);
    std::string s(n, '\0');
    is.read(s.data(), static_cast<std::streamsize>(n));
    if (!is)
        throw std::runtime_error(std::string("VectorModel::Load truncated reading ") + what);
    return s;
}

void WriteVec(std::ostream& os, std::span<const float> v)
{
    WriteRaw(os, static_cast<uint64_t>(v.size()));
    if (!v.empty())
        WriteBytes(os, v.data(), v.size() * sizeof(float));
}

std::vector<float> ReadVec(std::istream& is, const char* what)
{
    const uint64_t n = ReadRaw<uint64_t>(is, what);
    std::vector<float> v(static_cast<size_t>(n));
    if (n)
    {
        is.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
        if (!is)
            throw std::runtime_error(std::string("VectorModel::Load truncated reading ") + what);
    }
    return v;
}

void WriteNorm(std::ostream& os, const Normaliser& n)
{
    WriteRaw(os, n.Clip());
    WriteVec(os, n.Mean());
    WriteVec(os, n.Std());
}

Normaliser ReadNorm(std::istream& is)
{
    const float clip = ReadRaw<float>(is, "norm.clip");
    const auto mean = ReadVec(is, "norm.mean");
    const auto stdv = ReadVec(is, "norm.std");
    return Normaliser::FromState(mean, stdv, clip);
}

void WriteHead(std::ostream& os, std::string_view name, const Head& h)
{
    WriteString(os, name);
    WriteRaw(os, static_cast<uint8_t>(h.GetSign() == Head::Sign::Reward ? 1 : 0));
    WriteRaw(os, static_cast<uint8_t>(h.UsesZa() ? 1 : 0));
    WriteRaw(os, static_cast<uint64_t>(h.CodeSize()));
    const Head::Config cfg = h.GetConfig();
    WriteRaw(os, cfg.seed);
    WriteRaw(os, static_cast<uint64_t>(cfg.z_max));
    WriteRaw(os, static_cast<uint64_t>(cfg.gather_span));
    WriteRaw(os, static_cast<uint8_t>(cfg.tanh_last ? 1 : 0));
    WriteRaw(os, cfg.lr);
    WriteRaw(os, cfg.lr_min_frac);
    WriteRaw(os, static_cast<int32_t>(cfg.lr_decay_epochs));
    WriteRaw(os, static_cast<uint8_t>(cfg.restore_best ? 1 : 0));
    const LCNConfig nc = h.Net().Config();
    WriteRaw(os, static_cast<uint64_t>(nc.dim));
    WriteRaw(os, static_cast<uint64_t>(nc.z_max));
    WriteVec(os, h.Net().Weights());
}

std::pair<std::string, Head> ReadHead(std::istream& is)
{
    const std::string name = ReadString(is, "head.name");
    Head::Config cfg;
    cfg.sign = ReadRaw<uint8_t>(is, "head.sign") ? Head::Sign::Reward : Head::Sign::Cost;
    const bool uses_za = ReadRaw<uint8_t>(is, "head.uses_za") != 0;
    const size_t code = static_cast<size_t>(ReadRaw<uint64_t>(is, "head.code"));
    cfg.seed = ReadRaw<uint64_t>(is, "head.seed");
    cfg.z_max = static_cast<size_t>(ReadRaw<uint64_t>(is, "head.z_max"));
    cfg.gather_span = static_cast<size_t>(ReadRaw<uint64_t>(is, "head.gather_span"));
    cfg.tanh_last = ReadRaw<uint8_t>(is, "head.tanh_last") != 0;
    cfg.lr = ReadRaw<float>(is, "head.lr");
    cfg.lr_min_frac = ReadRaw<float>(is, "head.lr_min_frac");
    cfg.lr_decay_epochs = static_cast<int>(ReadRaw<int32_t>(is, "head.lr_decay_epochs"));
    cfg.restore_best = ReadRaw<uint8_t>(is, "head.restore_best") != 0;
    (void)ReadRaw<uint64_t>(is, "head.net_dim");
    const auto resolved_z = ReadRaw<uint64_t>(is, "head.net_z_max");
    cfg.z_max = static_cast<size_t>(resolved_z);
    const auto w = ReadVec(is, "head.weights");
    return {name, Head::FromState(cfg, uses_za, code, w)};
}

size_t CeilLog2(size_t n)
{
    size_t log = 0;
    while ((size_t{1} << log) < n)
        ++log;
    return log;
}

} // namespace

size_t MinDim(size_t obs_dim)
{
    if (obs_dim == 0)
        throw std::invalid_argument("MinDim obs_dim must be > 0");
    const size_t log = CeilLog2(obs_dim);
    return log > 6 ? log : 6;
}

size_t MinK(size_t act_dim)
{
    if (act_dim == 0)
        throw std::invalid_argument("MinK act_dim must be > 0");
    const size_t log = CeilLog2(act_dim);
    return log > 5 ? log : 5;
}

size_t RequireLastDim(size_t d, size_t limit, const char* what, const char* limit_name)
{
    if (d > limit)
        throw std::invalid_argument(std::string(what) + " last-dim " + std::to_string(d) +
                                    " > " + limit_name + " " + std::to_string(limit));
    return d;
}

std::unique_ptr<VectorModel> VectorModel::Create(std::unique_ptr<WorldModel> wm)
{
    if (!wm)
        throw std::invalid_argument("VectorModel::Create needs a WorldModel");
    auto vm = std::unique_ptr<VectorModel>(new VectorModel());
    vm->owned_ = std::move(wm);
    vm->wm_ = vm->owned_.get();
    vm->field_.resize(vm->wm_->FieldSize());
    vm->picture_.resize(vm->wm_->CodeSize());
    return vm;
}

std::unique_ptr<VectorModel> VectorModel::Attach(WorldModel& wm)
{
    auto vm = std::unique_ptr<VectorModel>(new VectorModel());
    vm->wm_ = &wm;
    vm->field_.resize(vm->wm_->FieldSize());
    vm->picture_.resize(vm->wm_->CodeSize());
    return vm;
}

void VectorModel::NoteObsDim(size_t d)
{
    if (obs_dim_ == 0)
        obs_dim_ = d;
    else if (obs_dim_ != d)
        throw std::invalid_argument("VectorModel obs_dim " + std::to_string(d) +
                                    " != " + std::to_string(obs_dim_));
}

void VectorModel::NoteActDim(size_t d)
{
    if (act_dim_ == 0)
        act_dim_ = d;
    else if (act_dim_ != d)
        throw std::invalid_argument("VectorModel act_dim " + std::to_string(d) +
                                    " != " + std::to_string(act_dim_));
}

void VectorModel::SetObsDim(size_t d)
{
    if (d == 0)
        throw std::invalid_argument("VectorModel obs_dim must be > 0");
    RequireLastDim(d, FieldSize(), "obs", "N");
    NoteObsDim(d);
}

void VectorModel::SetActDim(size_t d)
{
    if (d == 0)
        throw std::invalid_argument("VectorModel act_dim must be > 0");
    RequireLastDim(d, CodeSize(), "act", "code_size");
    NoteActDim(d);
}

void VectorModel::SetObsNormaliser(Normaliser n)
{
    RequireLastDim(n.Dim(), FieldSize(), "obs", "N");
    NoteObsDim(n.Dim());
    obs_norm_ = std::move(n);
}

void VectorModel::SetActNormaliser(Normaliser n)
{
    RequireLastDim(n.Dim(), CodeSize(), "act", "code_size");
    NoteActDim(n.Dim());
    act_norm_ = std::move(n);
}

void VectorModel::ClearObsNormaliser() { obs_norm_.reset(); }
void VectorModel::ClearActNormaliser() { act_norm_.reset(); }

const Normaliser* VectorModel::ObsNormaliser() const
{
    return obs_norm_ ? &*obs_norm_ : nullptr;
}

const Normaliser* VectorModel::ActNormaliser() const
{
    return act_norm_ ? &*act_norm_ : nullptr;
}

void VectorModel::SetActionBounds(std::span<const float> low, std::span<const float> high)
{
    if (low.size() != high.size() || low.empty())
        throw std::invalid_argument("VectorModel action bounds must be the same non-zero length");
    RequireLastDim(low.size(), CodeSize(), "act", "code_size");
    NoteActDim(low.size());
    action_low_.assign(low.begin(), low.end());
    action_high_.assign(high.begin(), high.end());
}

void VectorModel::ClearActionBounds()
{
    action_low_.clear();
    action_high_.clear();
}

void VectorModel::SetHead(std::string name, Head h)
{
    if (name.empty())
        throw std::invalid_argument("VectorModel Head name must not be empty");
    if (!h.Fitted())
        throw std::invalid_argument("VectorModel SetHead requires a fitted Head");
    heads_.insert_or_assign(std::move(name), std::move(h));
}

void VectorModel::RemoveHead(std::string_view name)
{
    auto it = heads_.find(std::string(name));
    if (it != heads_.end())
        heads_.erase(it);
}

Head* VectorModel::GetHead(std::string_view name)
{
    auto it = heads_.find(std::string(name));
    return it == heads_.end() ? nullptr : &it->second;
}

const Head* VectorModel::GetHead(std::string_view name) const
{
    auto it = heads_.find(std::string(name));
    return it == heads_.end() ? nullptr : &it->second;
}

void VectorModel::SetMeta(std::string key, std::string value)
{
    if (key.empty())
        throw std::invalid_argument("VectorModel meta key must not be empty");
    meta_[std::move(key)] = std::move(value);
}

std::string VectorModel::Meta(std::string_view key) const
{
    auto it = meta_.find(std::string(key));
    return it == meta_.end() ? std::string() : it->second;
}

const float* VectorModel::Encode(std::span<const float> obs, std::span<float> dst)
{
    RequireLastDim(obs.size(), FieldSize(), "obs", "N");
    NoteObsDim(obs.size());
    std::span<const float> src = obs;
    if (obs_norm_)
    {
        scratch_.resize(obs.size());
        obs_norm_->Apply(obs, scratch_);
        src = scratch_;
    }
    PaintStripes(src, field_);
    return wm_->Encode(field_, dst);
}

const float* VectorModel::EncodeAction(std::span<const float> a, std::span<float> dst)
{
    RequireLastDim(a.size(), CodeSize(), "act", "code_size");
    NoteActDim(a.size());
    std::span<const float> src = a;
    if (act_norm_)
    {
        scratch_.resize(a.size());
        act_norm_->Apply(a, scratch_);
        src = scratch_;
    }
    PaintStripes(src, picture_);
    return wm_->EncodeAction(picture_, dst);
}

const float* VectorModel::Predict(std::span<const float> z, std::span<const float> za)
{
    return wm_->Predict(z, za);
}

void VectorModel::Rollout(std::span<const float> z0, std::span<const float> actions,
                          std::span<float> out)
{
    if (act_dim_ == 0)
        throw std::invalid_argument("VectorModel::Rollout needs act_dim (encode an action or set bounds first)");
    if (actions.size() % act_dim_ != 0)
        throw std::invalid_argument("VectorModel::Rollout actions length must be a multiple of act_dim");
    const size_t h = actions.size() / act_dim_;
    const size_t c = CodeSize();
    action_codes_.resize(h * c);
    std::vector<float> za(c);
    for (size_t t = 0; t < h; ++t)
    {
        EncodeAction(actions.subspan(t * act_dim_, act_dim_), za);
        std::memcpy(action_codes_.data() + t * c, za.data(), c * sizeof(float));
    }
    wm_->Rollout(z0, action_codes_, out);
}

void VectorModel::Cost(std::span<const float> zs, size_t batch, size_t h1,
                       std::span<float> out, std::span<const float> goal_z) const
{
    const size_t c = CodeSize();
    if (batch == 0 || h1 == 0)
        throw std::invalid_argument("VectorModel::Cost needs batch > 0 and H+1 > 0");
    if (zs.size() != batch * h1 * c)
        throw std::invalid_argument("VectorModel::Cost zs length must be batch * (H+1) * code");
    if (out.size() != batch)
        throw std::invalid_argument("VectorModel::Cost out must be batch long");

    if (heads_.size() == 1)
    {
        heads_.begin()->second.PlanCost(zs, batch, h1, out);
        return;
    }
    if (goal_z.size() == c)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            const float* last = zs.data() + (b * h1 + (h1 - 1)) * c;
            double s = 0.0;
            for (size_t i = 0; i < c; ++i)
            {
                const double d = static_cast<double>(last[i]) - static_cast<double>(goal_z[i]);
                s += d * d;
            }
            out[b] = static_cast<float>(s);
        }
        return;
    }
    throw std::invalid_argument(
        "VectorModel::Cost needs a Head, or a goal_z of CodeSize()");
}

void VectorModel::Save(const std::filesystem::path& file) const
{
    std::ofstream os(file, std::ios::binary);
    if (!os)
        throw std::runtime_error("VectorModel::Save cannot open " + file.string());

    os.write(kMagic, sizeof(kMagic));
    WriteRaw(os, kFileVersion);
    WriteRaw(os, static_cast<uint64_t>(obs_dim_));
    WriteRaw(os, static_cast<uint64_t>(act_dim_));
    WriteRaw(os, static_cast<uint8_t>(obs_norm_ ? 1 : 0));
    WriteRaw(os, static_cast<uint8_t>(act_norm_ ? 1 : 0));
    WriteRaw(os, static_cast<uint8_t>(HasActionBounds() ? 1 : 0));
    WriteRaw(os, static_cast<uint32_t>(heads_.size()));
    WriteRaw(os, static_cast<uint32_t>(meta_.size()));

    if (obs_norm_)
        WriteNorm(os, *obs_norm_);
    if (act_norm_)
        WriteNorm(os, *act_norm_);
    if (HasActionBounds())
    {
        WriteVec(os, action_low_);
        WriteVec(os, action_high_);
    }
    for (const auto& [name, h] : heads_)
        WriteHead(os, name, h);
    for (const auto& [k, v] : meta_)
    {
        WriteString(os, k);
        WriteString(os, v);
    }

    std::ostringstream blob(std::ios::binary);
    wm_->Save(blob);
    const std::string bytes = blob.str();
    WriteRaw(os, static_cast<uint64_t>(bytes.size()));
    WriteBytes(os, bytes.data(), bytes.size());

    if (!os)
        throw std::runtime_error("VectorModel::Save write failed for " + file.string());
}

std::unique_ptr<VectorModel> VectorModel::Load(const std::filesystem::path& file)
{
    std::ifstream is(file, std::ios::binary);
    if (!is)
        throw std::runtime_error("VectorModel::Load cannot open " + file.string());

    char magic[4];
    is.read(magic, sizeof(magic));
    if (!is)
        throw std::runtime_error("VectorModel::Load truncated reading magic in " + file.string());

    if (std::memcmp(magic, kWorldMagic, 4) == 0)
    {
        is.seekg(0);
        auto wm = WorldModel::Load(is, file.string());
        return Create(std::move(wm));  // owns the loaded WorldModel
    }
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("VectorModel::Load bad magic in " + file.string());

    const uint32_t version = ReadRaw<uint32_t>(is, "version");
    if (version != kFileVersion)
        throw std::runtime_error("VectorModel::Load unsupported version " +
                                 std::to_string(version) + " in " + file.string());

    auto vm = std::unique_ptr<VectorModel>(new VectorModel());
    vm->obs_dim_ = static_cast<size_t>(ReadRaw<uint64_t>(is, "obs_dim"));
    vm->act_dim_ = static_cast<size_t>(ReadRaw<uint64_t>(is, "act_dim"));
    const bool has_obs = ReadRaw<uint8_t>(is, "has_obs_norm") != 0;
    const bool has_act = ReadRaw<uint8_t>(is, "has_act_norm") != 0;
    const bool has_bounds = ReadRaw<uint8_t>(is, "has_bounds") != 0;
    const uint32_t n_heads = ReadRaw<uint32_t>(is, "n_heads");
    const uint32_t n_meta = ReadRaw<uint32_t>(is, "n_meta");

    if (has_obs)
        vm->obs_norm_ = ReadNorm(is);
    if (has_act)
        vm->act_norm_ = ReadNorm(is);
    if (has_bounds)
    {
        vm->action_low_ = ReadVec(is, "action_low");
        vm->action_high_ = ReadVec(is, "action_high");
    }
    for (uint32_t i = 0; i < n_heads; ++i)
    {
        auto [name, h] = ReadHead(is);
        vm->heads_.insert_or_assign(std::move(name), std::move(h));
    }
    for (uint32_t i = 0; i < n_meta; ++i)
    {
        const std::string k = ReadString(is, "meta.key");
        const std::string v = ReadString(is, "meta.val");
        vm->meta_[k] = v;
    }

    const uint64_t n_bytes = ReadRaw<uint64_t>(is, "wm_bytes");
    std::string bytes(static_cast<size_t>(n_bytes), '\0');
    is.read(bytes.data(), static_cast<std::streamsize>(n_bytes));
    if (!is)
        throw std::runtime_error("VectorModel::Load truncated reading WorldModel in " +
                                 file.string());
    std::istringstream blob(bytes, std::ios::binary);
    vm->owned_ = WorldModel::Load(blob, file.string() + ":world");
    vm->wm_ = vm->owned_.get();
    vm->field_.resize(vm->wm_->FieldSize());
    vm->picture_.resize(vm->wm_->CodeSize());
    return vm;
}
