// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Solver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

void DenseSolve(std::span<double> A, size_t n, std::span<double> B, size_t nrhs)
{
    if (n == 0)
        throw std::invalid_argument("DenseSolve n must be > 0");
    if (A.size() != n * n)
        throw std::invalid_argument("DenseSolve A must be n by n");
    if (nrhs == 0 || B.size() != n * nrhs)
        throw std::invalid_argument("DenseSolve B must be n by nrhs");

    std::vector<size_t> piv(n);
    for (size_t i = 0; i < n; ++i)
        piv[i] = i;

    for (size_t k = 0; k < n; ++k)
    {
        size_t best = k;
        double best_abs = std::fabs(A[k * n + k]);
        for (size_t i = k + 1; i < n; ++i)
        {
            const double a = std::fabs(A[i * n + k]);
            if (a > best_abs)
            {
                best_abs = a;
                best = i;
            }
        }
        if (!(best_abs > 0.0))
            throw std::invalid_argument("DenseSolve singular matrix");
        if (best != k)
        {
            for (size_t j = 0; j < n; ++j)
                std::swap(A[k * n + j], A[best * n + j]);
            for (size_t j = 0; j < nrhs; ++j)
                std::swap(B[k * nrhs + j], B[best * nrhs + j]);
            std::swap(piv[k], piv[best]);
        }
        const double akk = A[k * n + k];
        for (size_t i = k + 1; i < n; ++i)
        {
            const double f = A[i * n + k] / akk;
            A[i * n + k] = f;
            for (size_t j = k + 1; j < n; ++j)
                A[i * n + j] -= f * A[k * n + j];
            for (size_t j = 0; j < nrhs; ++j)
                B[i * nrhs + j] -= f * B[k * nrhs + j];
        }
    }

    for (size_t k = n; k-- > 0;)
    {
        const double akk = A[k * n + k];
        for (size_t j = 0; j < nrhs; ++j)
        {
            double s = B[k * nrhs + j];
            for (size_t i = k + 1; i < n; ++i)
                s -= A[k * n + i] * B[i * nrhs + j];
            B[k * nrhs + j] = s / akk;
        }
    }
}
