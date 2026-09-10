// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/// @brief Persistent workers. Run(fn) calls fn(0) .. fn(n-1) in parallel
/// and blocks until they all finish. n=1 runs fn(0) on the calling thread.
///
/// One instance is not thread-safe for concurrent Run calls.
class ThreadPool
{
public:
    /// @brief Start n workers. If a thread cannot be started, the workers
    /// already running are stopped and joined before the exception
    /// (std::system_error) propagates, so no joinable thread is destroyed.
    explicit ThreadPool(size_t n);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    [[nodiscard]] size_t Size() const { return n_; }

    /// @brief Call @p fn(worker_id) once per worker, worker_id in [0, Size()).
    /// Rethrows the first exception thrown in a worker.
    void Run(std::function<void(size_t)> fn);

private:
    void Worker(size_t id);

    /// Set stop, wake every worker, and join them all. Idempotent.
    void Shutdown() noexcept;

    size_t n_ = 1;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::function<void(size_t)> fn_;
    std::exception_ptr eptr_;
    size_t gen_ = 0;
    size_t remaining_ = 0;
    bool stop_ = false;
};
