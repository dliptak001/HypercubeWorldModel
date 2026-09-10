#include "Walker.h"

#include <stdexcept>

Walker::Walker(const TerrainMap& map, int r, int c, Heading heading)
    : map_(&map)
    , r_(r)
    , c_(c)
    , heading_(heading)
{
    if (!map.CropFits(r, c))
        throw std::invalid_argument("Walker start crop would leave the map");
    if (map.IsBarrier(r, c))
        throw std::invalid_argument("Walker start is a barrier");
}

void Walker::FillView(std::span<float> out) const
{
    map_->FillView(r_, c_, out);
}

bool Walker::Legal(Heading h) const
{
    const auto [dr, dc] = Delta(h);
    const int nr = r_ + dr;
    const int nc = c_ + dc;
    if (!map_->CropFits(nr, nc))
        return false;
    if (map_->IsBarrier(nr, nc))
        return false;
    return true;
}

Heading Walker::Intended(std::mt19937_64& rng) const
{
    if (!map_->GoalInCrop(r_, c_))
        return heading_;

    const int dr = map_->GoalR() - r_;
    const int dc = map_->GoalC() - c_;
    if (dr == 0 && dc == 0)
        return heading_;

    Heading cand[4];
    int n = 0;
    if (dr < 0)
        cand[n++] = Heading::North;
    if (dr > 0)
        cand[n++] = Heading::South;
    if (dc < 0)
        cand[n++] = Heading::West;
    if (dc > 0)
        cand[n++] = Heading::East;
    for (int i = 0; i < n; ++i)
        if (cand[i] == heading_)
            return heading_;
    if (n <= 0)
        return heading_;
    if (n == 1)
        return cand[0];
    std::uniform_int_distribution<int> pick(0, n - 1);
    return cand[pick(rng)];
}

Heading Walker::Recover(Heading intended, std::mt19937_64& rng, bool& turned) const
{
    if (Legal(intended))
    {
        turned = false;
        return intended;
    }
    turned = true;
    const Heading L = TurnLeft(heading_);
    const Heading R = TurnRight(heading_);
    const bool l_ok = Legal(L);
    const bool r_ok = Legal(R);
    if (l_ok && r_ok)
    {
        std::uniform_int_distribution<int> bit(0, 1);
        return bit(rng) ? L : R;
    }
    if (l_ok)
        return L;
    if (r_ok)
        return R;
    const Heading U = TurnAbout(heading_);
    if (Legal(U))
        return U;
    return intended;
}

bool Walker::Step(std::mt19937_64& rng, Heading& executed, bool& turned)
{
    const Heading intended = Intended(rng);
    executed = Recover(intended, rng, turned);
    if (!Legal(executed))
        return false;
    heading_ = executed;
    const auto [dr, dc] = Delta(executed);
    r_ += dr;
    c_ += dc;
    return true;
}

Walk RecordWalk(const TerrainMap& map, int r, int c, Heading heading,
                int n_steps, std::mt19937_64& rng)
{
    if (n_steps < 1)
        throw std::invalid_argument("RecordWalk n_steps");
    Walker w(map, r, c, heading);
    Walk out;
    const size_t n = static_cast<size_t>(map.View()) * static_cast<size_t>(map.View());
    out.frames.reserve(static_cast<size_t>(n_steps) + 1);
    out.actions.reserve(static_cast<size_t>(n_steps));
    out.turned.reserve(static_cast<size_t>(n_steps));
    out.changed.reserve(static_cast<size_t>(n_steps));

    auto push_frame = [&]()
    {
        std::vector<float> x(n);
        w.FillView(x);
        out.frames.push_back(std::move(x));
        if (map.GoalInCrop(w.Row(), w.Col()))
            out.saw_goal = true;
        if (w.Row() == map.GoalR() && w.Col() == map.GoalC())
            out.reached = true;
    };

    push_frame();
    for (int s = 0; s < n_steps; ++s)
    {
        const Heading prev = w.GetHeading();
        Heading exec{};
        bool turned = false;
        if (!w.Step(rng, exec, turned))
            break;
        out.actions.push_back(exec);
        out.turned.push_back(turned ? 1 : 0);
        out.changed.push_back(exec != prev ? 1 : 0);
        push_frame();
    }
    return out;
}
