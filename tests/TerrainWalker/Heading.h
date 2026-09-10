#pragma once

#include <utility>

enum class Heading { North = 0, East = 1, South = 2, West = 3 };

inline constexpr int kHeadingCount = 4;

inline int Index(Heading h)
{
    return static_cast<int>(h);
}

inline Heading HeadingFromIndex(int i)
{
    return static_cast<Heading>(i);
}

inline std::pair<int, int> Delta(Heading h)
{
    switch (h)
    {
    case Heading::North: return {-1, 0};
    case Heading::East:  return {0, 1};
    case Heading::South: return {1, 0};
    case Heading::West:  return {0, -1};
    }
    return {0, 0};
}

inline Heading TurnLeft(Heading h)
{
    switch (h)
    {
    case Heading::North: return Heading::West;
    case Heading::West:  return Heading::South;
    case Heading::South: return Heading::East;
    case Heading::East:  return Heading::North;
    }
    return h;
}

inline Heading TurnRight(Heading h)
{
    switch (h)
    {
    case Heading::North: return Heading::East;
    case Heading::East:  return Heading::South;
    case Heading::South: return Heading::West;
    case Heading::West:  return Heading::North;
    }
    return h;
}

inline Heading TurnAbout(Heading h)
{
    switch (h)
    {
    case Heading::North: return Heading::South;
    case Heading::South: return Heading::North;
    case Heading::East:  return Heading::West;
    case Heading::West:  return Heading::East;
    }
    return h;
}
