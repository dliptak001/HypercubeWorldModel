// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include "Head.h"
#include "Normaliser.h"
#include "WorldModel.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// @brief Floor view-cube dim for a vector of this length. Not a config.
/// max(6, ceil(log2(obs_dim))).
/// @throws std::invalid_argument if @p obs_dim is 0.
[[nodiscard]] size_t MinDim(size_t obs_dim);

/// @brief Floor action-cube dim for a vector of this length. Not a config.
/// max(5, ceil(log2(act_dim))).
/// @throws std::invalid_argument if @p act_dim is 0.
[[nodiscard]] size_t MinK(size_t act_dim);

/// @brief Raise if @p d exceeds @p limit, naming both numbers.
/// @return @p d.
/// @throws std::invalid_argument with "{what} last-dim {d} > {limit_name} {limit}".
size_t RequireLastDim(size_t d, size_t limit, const char* what, const char* limit_name);

/// @brief Vector front-end on a WorldModel: short obs/act in, codes out.
///
/// Always paints with PaintStripes. Optional obs/act Normaliser. Rollout
/// takes raw actions (paint, EncodeAction the block once, WorldModel::Rollout
/// on codes). Named Heads, optional action bounds. Hard capacity checks:
/// obs last-dim vs N, act last-dim vs code_size, both numbers in the error.
///
/// A host that already has a field uses WorldModel. Nothing here is added
/// to WorldModel. Save writes a new magic; HWM1 is unchanged. Load of an
/// HWM1 file constructs a VectorModel with no attachments.
///
/// Non-copyable; obtain instances via Create or Load. One instance is not
/// thread-safe for concurrent public calls (it uses the WorldModel).
class VectorModel
{
public:
    static constexpr char kMagic[4] = {'H', 'V', 'M', '1'};
    static constexpr uint32_t kFileVersion = 1;

    /// Take ownership of @p wm.
    static std::unique_ptr<VectorModel> Create(std::unique_ptr<WorldModel> wm);

    /// Non-owning view of @p wm. The WorldModel must outlive this
    /// VectorModel (Python keep_alive enforces that).
    static std::unique_ptr<VectorModel> Attach(WorldModel& wm);

    /// @brief Read a VectorModel written by @ref Save, or a bare HWM1 as
    /// a wm-only VectorModel.
    /// @throws std::runtime_error on file trouble or bad magic/version.
    static std::unique_ptr<VectorModel> Load(const std::filesystem::path& file);

    /// @brief Write this VectorModel: new magic, then attachments, then a
    /// length-prefixed WorldModel HWM1 payload. Optional attachments are
    /// omitted when absent. Does not change HWM1.
    void Save(const std::filesystem::path& file) const;

    VectorModel(const VectorModel&) = delete;
    VectorModel& operator=(const VectorModel&) = delete;

    [[nodiscard]] WorldModel& World() { return *wm_; }
    [[nodiscard]] const WorldModel& World() const { return *wm_; }

    void SetObsNormaliser(Normaliser n);
    void SetActNormaliser(Normaliser n);
    void ClearObsNormaliser();
    void ClearActNormaliser();
    [[nodiscard]] const Normaliser* ObsNormaliser() const;
    [[nodiscard]] const Normaliser* ActNormaliser() const;

    void SetActionBounds(std::span<const float> low, std::span<const float> high);
    void ClearActionBounds();
    [[nodiscard]] bool HasActionBounds() const { return !action_low_.empty(); }
    [[nodiscard]] std::span<const float> ActionLow() const { return action_low_; }
    [[nodiscard]] std::span<const float> ActionHigh() const { return action_high_; }

    void SetHead(std::string name, Head h);
    void RemoveHead(std::string_view name);
    [[nodiscard]] Head* GetHead(std::string_view name);
    [[nodiscard]] const Head* GetHead(std::string_view name) const;
    [[nodiscard]] const std::map<std::string, Head>& Heads() const { return heads_; }

    void SetMeta(std::string key, std::string value);
    [[nodiscard]] std::string Meta(std::string_view key) const;
    [[nodiscard]] const std::map<std::string, std::string>& MetaMap() const { return meta_; }

    [[nodiscard]] size_t ObsDim() const { return obs_dim_; }
    [[nodiscard]] size_t ActDim() const { return act_dim_; }
    void SetObsDim(size_t d);
    void SetActDim(size_t d);

    [[nodiscard]] size_t FieldSize() const { return wm_->FieldSize(); }
    [[nodiscard]] size_t CodeSize() const { return wm_->CodeSize(); }

    /// One observation vector. Writes CodeSize() into @p dst.
    const float* Encode(std::span<const float> obs, std::span<float> dst);

    /// One raw action vector. Writes CodeSize() into @p dst.
    const float* EncodeAction(std::span<const float> a, std::span<float> dst);

    const float* Predict(std::span<const float> z, std::span<const float> za);

    /// Raw-action rollout. @p z0 is CodeSize(); @p actions is H × act_dim
    /// laid end to end; @p out is (H+1) × CodeSize(). H may be 0.
    void Rollout(std::span<const float> z0, std::span<const float> actions,
                 std::span<float> out);

    /// Score a rollout. If there is exactly one Head, uses its PlanCost.
    /// Else if @p goal_z is CodeSize() long, L2 of the last code vs the
    /// goal. @p zs is (batch × (H+1) × code); @p out is batch long.
    void Cost(std::span<const float> zs, size_t batch, size_t h1,
              std::span<float> out, std::span<const float> goal_z = {}) const;

private:
    VectorModel() = default;

    void NoteObsDim(size_t d);
    void NoteActDim(size_t d);

    std::unique_ptr<WorldModel> owned_;
    WorldModel* wm_ = nullptr;
    std::vector<float> field_;
    std::vector<float> picture_;
    std::vector<float> scratch_;
    std::vector<float> action_codes_;
    std::vector<float> action_low_, action_high_;
    std::optional<Normaliser> obs_norm_;
    std::optional<Normaliser> act_norm_;
    std::map<std::string, Head> heads_;
    std::map<std::string, std::string> meta_;
    size_t obs_dim_ = 0;
    size_t act_dim_ = 0;
};
