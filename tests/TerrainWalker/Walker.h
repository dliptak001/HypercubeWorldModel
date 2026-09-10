#pragma once

#include "Heading.h"
#include "Map.h"

#include <random>
#include <span>
#include <vector>

/// One recorded wander-then-acquire walk on a single map.
struct Walk
{
    std::vector<std::vector<float>> frames; // N floats each; length = steps+1
    std::vector<Heading> actions;           // executed cardinal; length = steps
    std::vector<char> turned;               // 1 if the 90°/180° encounter rule
    std::vector<char> changed;              // 1 if executed heading != heading before the step
    bool saw_goal = false;
    bool reached = false;
};

/// Camera center on a map. Wander until the goal is in view, then step
/// toward it. Illegal heading → turn 90° and still step.
class Walker
{
public:
    Walker(const TerrainMap& map, int r, int c, Heading heading);

    [[nodiscard]] int Row() const { return r_; }
    [[nodiscard]] int Col() const { return c_; }
    [[nodiscard]] Heading GetHeading() const { return heading_; }

    void FillView(std::span<float> out) const;

    /// One executed step. False if boxed in (no legal cardinal).
    bool Step(std::mt19937_64& rng, Heading& executed, bool& turned);

private:
    [[nodiscard]] bool Legal(Heading h) const;
    [[nodiscard]] Heading Intended(std::mt19937_64& rng) const;
    Heading Recover(Heading intended, std::mt19937_64& rng, bool& turned) const;

    const TerrainMap* map_;
    int r_ = 0;
    int c_ = 0;
    Heading heading_ = Heading::North;
};

Walk RecordWalk(const TerrainMap& map, int r, int c, Heading heading,
                int n_steps, std::mt19937_64& rng);
