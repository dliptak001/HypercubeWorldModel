#pragma once

#include <random>
#include <span>
#include <vector>

/// 64×64 elevation grid with barriers and one goal. The 16×16 view is
/// cropped from here; this object does not walk.
class TerrainMap
{
public:
    /// Elevations from a few 2D sines, rescaled to [-0.5, 0.5]. Barriers
    /// at random (not on the goal). Goal in the legal pose rectangle.
    static TerrainMap Draw(int size, int view, float barrier_frac,
                           int sine_terms, float cycles_min, float cycles_max,
                           float amp_min, float amp_max, std::mt19937_64& rng);

    [[nodiscard]] int Size() const { return size_; }
    [[nodiscard]] int View() const { return view_; }
    [[nodiscard]] int GoalR() const { return goal_r_; }
    [[nodiscard]] int GoalC() const { return goal_c_; }

    /// Inclusive pose range so the view stays fully on-map.
    [[nodiscard]] int PoseMin() const { return view_ / 2; }
    [[nodiscard]] int PoseMax() const { return size_ - view_ + view_ / 2; }

    [[nodiscard]] bool CropFits(int r, int c) const;
    [[nodiscard]] bool IsBarrier(int r, int c) const;
    [[nodiscard]] bool GoalInCrop(int r, int c) const;

    /// 16×16 row-major. Goal +1, barrier −1, else elevation.
    void FillView(int r, int c, std::span<float> out) const;

    /// Free legal-pose cell, not a barrier, not the goal.
    bool SampleFreePose(std::mt19937_64& rng, int& r, int& c) const;

private:
    int size_ = 0;
    int view_ = 0;
    int goal_r_ = 0;
    int goal_c_ = 0;
    std::vector<float> cells_;
};
