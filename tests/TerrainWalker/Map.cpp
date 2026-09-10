#include "Map.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

bool TerrainMap::CropFits(int r, int c) const
{
    return r >= PoseMin() && r <= PoseMax() && c >= PoseMin() && c <= PoseMax();
}

bool TerrainMap::IsBarrier(int r, int c) const
{
    if (r < 0 || c < 0 || r >= size_ || c >= size_)
        return true;
    return cells_[static_cast<size_t>(r * size_ + c)] == -1.f;
}

bool TerrainMap::GoalInCrop(int r, int c) const
{
    const int r0 = r - view_ / 2;
    const int c0 = c - view_ / 2;
    const int r1 = r0 + view_ - 1;
    const int c1 = c0 + view_ - 1;
    return goal_r_ >= r0 && goal_r_ <= r1 && goal_c_ >= c0 && goal_c_ <= c1;
}

void TerrainMap::FillView(int r, int c, std::span<float> out) const
{
    const size_t n = static_cast<size_t>(view_) * static_cast<size_t>(view_);
    if (out.size() != n)
        throw std::invalid_argument("TerrainMap::FillView out must be view*view");
    if (!CropFits(r, c))
        throw std::invalid_argument("TerrainMap::FillView crop would leave the map");
    const int r0 = r - view_ / 2;
    const int c0 = c - view_ / 2;
    for (int i = 0; i < view_; ++i)
    {
        for (int j = 0; j < view_; ++j)
        {
            const int rr = r0 + i;
            const int cc = c0 + j;
            float v = cells_[static_cast<size_t>(rr * size_ + cc)];
            if (rr == goal_r_ && cc == goal_c_)
                v = 1.f;
            out[static_cast<size_t>(i * view_ + j)] = v;
        }
    }
}

bool TerrainMap::SampleFreePose(std::mt19937_64& rng, int& r, int& c) const
{
    std::uniform_int_distribution<int> pose(PoseMin(), PoseMax());
    const int tries = size_ * size_;
    for (int t = 0; t < tries; ++t)
    {
        const int rr = pose(rng);
        const int cc = pose(rng);
        if (IsBarrier(rr, cc))
            continue;
        if (rr == goal_r_ && cc == goal_c_)
            continue;
        r = rr;
        c = cc;
        return true;
    }
    return false;
}

TerrainMap TerrainMap::Draw(int size, int view, float barrier_frac,
                            int sine_terms, float cycles_min, float cycles_max,
                            float amp_min, float amp_max, std::mt19937_64& rng)
{
    if (size < 2 || view < 2 || view > size)
        throw std::invalid_argument("TerrainMap::Draw size/view");
    if (sine_terms < 1)
        throw std::invalid_argument("TerrainMap::Draw sine_terms");
    TerrainMap m;
    m.size_ = size;
    m.view_ = view;
    m.cells_.assign(static_cast<size_t>(size) * static_cast<size_t>(size), 0.f);

    std::uniform_real_distribution<float> cycles(cycles_min, cycles_max);
    std::uniform_real_distribution<float> phase(0.f, 2.f * std::numbers::pi_v<float>);
    std::uniform_real_distribution<float> amp(amp_min, amp_max);
    const float s = static_cast<float>(size);
    for (int t = 0; t < sine_terms; ++t)
    {
        const float fx = cycles(rng);
        const float fy = cycles(rng);
        const float ph = phase(rng);
        const float a = amp(rng);
        for (int r = 0; r < size; ++r)
        {
            for (int c = 0; c < size; ++c)
            {
                const float ang = 2.f * std::numbers::pi_v<float>
                    * (fx * static_cast<float>(r) + fy * static_cast<float>(c)) / s
                    + ph;
                m.cells_[static_cast<size_t>(r * size + c)] += a * std::sin(ang);
            }
        }
    }

    float lo = m.cells_[0];
    float hi = m.cells_[0];
    for (float v : m.cells_)
    {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float span = hi - lo;
    if (span > 0.f)
    {
        for (float& v : m.cells_)
            v = (v - lo) / span - 0.5f;
    }
    else
    {
        for (float& v : m.cells_)
            v = 0.f;
    }

    std::uniform_int_distribution<int> pose(m.PoseMin(), m.PoseMax());
    m.goal_r_ = pose(rng);
    m.goal_c_ = pose(rng);

    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int r = 0; r < size; ++r)
    {
        for (int c = 0; c < size; ++c)
        {
            if (r == m.goal_r_ && c == m.goal_c_)
                continue;
            if (u(rng) < barrier_frac)
                m.cells_[static_cast<size_t>(r * size + c)] = -1.f;
        }
    }
    return m;
}
