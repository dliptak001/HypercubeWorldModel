#pragma once

#include "Heading.h"

#include <span>

/// Cardinal as a rows × cols field, row-major. Half-plane of ones toward
/// the heading, zeros elsewhere: north is the top half of the rows,
/// south the bottom half, west the left half of the columns, east the
/// right half. Fat enough to survive EncodeAction. For the WorldModel
/// action cube, rows × cols is the k-face length 2ᵏ.
struct ActionField
{
    static void Paint(Heading h, int rows, int cols, std::span<float> field);
};
