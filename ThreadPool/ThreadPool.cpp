// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t n)
    : n_(n < 1 ? 1 : n)
{
    if (n_ == 1)
        return;
    threads_.reserve(n_);
    try
    {
        for (size_t i = 0; i < n_; ++i)
            threads_.emplace_back([this, i] { Worker(i); });
    }
    catch (...)
    {
        // A later std::thread failed to start. The earlier ones are already
        // running Worker on this half-built pool; stop and join them before
        // the vector destructor would std::terminate on a joinable thread.
        Shutdown();
        throw;
    }
}

ThreadPool::~ThreadPool()
{
    Shutdown();
}

void ThreadPool::Shutdown() noexcept
{
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto& th : threads_)
        if (th.joinable())
            th.join();
}

void ThreadPool::Run(std::function<void(size_t)> fn)
{
    if (n_ == 1)
    {
        fn(0);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        fn_ = std::move(fn);
        eptr_ = nullptr;
        remaining_ = n_;
        ++gen_;
    }
    cv_work_.notify_all();

    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [this] { return remaining_ == 0; });
    fn_ = nullptr;
    if (eptr_)
        std::rethrow_exception(eptr_);
}

void ThreadPool::Worker(size_t id)
{
    size_t seen = 0;
    for (;;)
    {
        std::unique_lock<std::mutex> lock(mu_);
        cv_work_.wait(lock, [&] { return stop_ || gen_ != seen; });
        if (stop_ && gen_ == seen)
            return;
        seen = gen_;
        auto fn = fn_;
        lock.unlock();
        try
        {
            if (fn)
                fn(id);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> g(mu_);
            if (!eptr_)
                eptr_ = std::current_exception();
        }
        lock.lock();
        if (--remaining_ == 0)
            cv_done_.notify_one();
    }
}
