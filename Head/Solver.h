// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstddef>
#include <span>

/// @brief Solve A X = B for square A (n × n).
///
/// A is row-major n×n and is overwritten. B is row-major n × nrhs and
/// is overwritten with X. Partial-pivoted Gaussian elimination; no
/// extra dependency. Used by Head::Fit and LinearR2.
/// @throws std::invalid_argument if n is 0, a span is the wrong length,
///         or A is singular.
void DenseSolve(std::span<double> A, size_t n, std::span<double> B, size_t nrhs);
