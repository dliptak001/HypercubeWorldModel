// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Predictor.h"

#include <stdexcept>

std::unique_ptr<Predictor> Predictor::Create(const PredictorConfig& cfg)
{
    return std::unique_ptr<Predictor>(new Predictor(cfg));
}

Predictor::Predictor(const PredictorConfig& cfg)
    : cfg_(cfg)
{
    net_ = LCN::Create(LCNConfig{.dim = cfg.dim,
                                 .seed = cfg.seed,
                                 .z_max = cfg.z_max,
                                 .gather_span = cfg.gather_span,
                                 .tanh_last = cfg.tanh_last});
    training_ = std::make_unique<LCNTraining>(*net_, cfg.training);
    cfg_.z_max = net_->ZMax();
    n_ = net_->N();
}

const float* Predictor::Predict(std::span<const float> z)
{
    if (z.size() != n_)
        throw std::invalid_argument("Predictor::Predict z must be Size() long");
    net_->Forward(z);
    return net_->Output().data();
}

void Predictor::BeginBatch()
{
    training_->ZeroGrad();
}

float Predictor::Accumulate(std::span<const float> z, std::span<const float> next)
{
    if (z.size() != n_)
        throw std::invalid_argument("Predictor::Accumulate z must be Size() long");
    if (next.empty() || next.size() > n_)
        throw std::invalid_argument(
            "Predictor::Accumulate next must be 1 .. Size() long");
    net_->Forward(z);
    const float loss = training_->Loss(next);
    training_->Backward();
    return loss;
}

void Predictor::EndBatch()
{
    training_->Adam();
}

void Predictor::SetEpoch(int epoch, int num_epochs)
{
    training_->SetEpoch(epoch, num_epochs);
}

void Predictor::Observe(float metric, int epoch)
{
    training_->Observe(metric, epoch);
}

void Predictor::RestoreBest()
{
    training_->RestoreBest();
}

void Predictor::ResetTraining()
{
    training_->Reset();
}

const std::vector<float>& Predictor::Weights() const
{
    return net_->Weights();
}

void Predictor::LoadWeights(std::span<const float> w)
{
    net_->LoadWeights(w);
}

const std::vector<float>& Predictor::Grad() const
{
    return training_->Grad();
}

void Predictor::AddGrad(std::span<const float> g)
{
    training_->AddGrad(g);
}
