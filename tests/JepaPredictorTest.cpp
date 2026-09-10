// Sequential-window JEPA: does Predictor map E(x_t) → E(x_{t+1})?
//
// One two-sine stream, cut into consecutive windows of length N.
// Encode each window; train Predictor on adjacent k-faces.
// Decoder is not used.

#include "Encoder.h"
#include "Predictor.h"
#include "report_config.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <span>
#include <vector>

// =============================================================================
// Shared cube
// =============================================================================

static constexpr size_t kDim = 8; // N = 2^dim
static constexpr size_t kSubcubeDim = 7; // k < dim; Predictor lives on this face

// =============================================================================
// Encoder configuration — primary knobs (edit here)
// =============================================================================

static EncoderConfig MakeEncoderConfig()
{
    EncoderConfig cfg;
    cfg.dim = kDim;
    cfg.seed = 1; // weight draw
    cfg.spectral_radius = 0.999f;
    cfg.leak_rate = 0.025;
    cfg.input_scaling = 0.8f;
    cfg.history_depth = 8; // M
    cfg.passes = 2*kDim; // T; 0 → T = N
    cfg.ic_seed = 2; // start state s0
    return cfg;
}

// =============================================================================
// Predictor configuration — primary knobs (edit here)
// =============================================================================

static PredictorConfig MakePredictorConfig()
{
    PredictorConfig cfg;
    cfg.dim = kSubcubeDim; // latent cube, not Encoder dim
    cfg.z_max = 3 * kSubcubeDim; // 0 → dim; dim = antipodal reach
    cfg.gather_span = 5;
    cfg.tanh_last = true;
    cfg.seed = 3; // LCN weight draw

    cfg.training.lr = 0.03f;
    cfg.training.lr_min_frac = 0.05f;
    cfg.training.lr_decay_epochs = 0; // 0 → kEpochs
    cfg.training.restore_best = true;
    cfg.training.beta1 = 0.9f;
    cfg.training.beta2 = 0.999f;
    cfg.training.eps = 1e-8f;
    return cfg;
}

// =============================================================================
// Task parameters (not part of EncoderConfig / PredictorConfig)
// =============================================================================

static constexpr int kTrain = 1024; // training windows (pairs = kTrain - 1)
static constexpr int kVal = kTrain / 2; // validation windows
static constexpr int kTest = kTrain / 2; // held-out windows
static constexpr int kEpochs = 100; // predictor training epochs
static constexpr size_t kHop = size_t{1} << kDim; // samples between window starts
static constexpr uint64_t kDataSeed = 4; // sine-stream generator

static constexpr int kSineTerms = 2;
static constexpr float kCyclesMin = 0.7f;
static constexpr float kCyclesMax = 7.3f;
static constexpr float kAmpMin = 0.3f;
static constexpr float kAmpMax = 1.0f;

// =============================================================================
// Helpers
// =============================================================================

static int Fail(const char* what)
{
    std::printf("FAIL: %s\n", what);
    return 1;
}

static float MeanSquare(std::span<const float> a, std::span<const float> b)
{
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s / static_cast<float>(a.size());
}

static float MeanSquare(std::span<const float> a)
{
    float s = 0.f;
    for (const float x : a)
        s += x * x;
    return s / static_cast<float>(a.size());
}

static double MeanAbs(std::span<const float> x)
{
    if (x.empty())
        return 0.0;
    double a = 0.0;
    for (float v : x)
        a += std::fabs(static_cast<double>(v));
    return a / static_cast<double>(x.size());
}

struct SineTerm
{
    float cycles = 0.f;
    float phase = 0.f;
    float amp = 0.f;
};

static void DrawSines(std::span<SineTerm> terms, std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> cycles(kCyclesMin, kCyclesMax);
    std::uniform_real_distribution<float> phase(0.f, 2.f * std::numbers::pi_v<float>);
    std::uniform_real_distribution<float> amp(kAmpMin, kAmpMax);
    for (SineTerm& t : terms)
    {
        t.cycles = cycles(rng);
        t.phase = phase(rng);
        t.amp = amp(rng);
    }
}

// One long 1D stream: same two sines for every sample. Cycle count is
// relative to a window of length N, matching CompressionTest.
static void FillSineStream(std::span<float> stream, std::span<const SineTerm> terms,
                           size_t window)
{
    const float n = static_cast<float>(window);
    for (float& v : stream)
        v = 0.f;
    for (const SineTerm& t : terms)
    {
        for (size_t i = 0; i < stream.size(); ++i)
            stream[i] += t.amp * std::sin(2.f * std::numbers::pi_v<float> * t.cycles
                * static_cast<float>(i) / n + t.phase);
    }
}

struct Split
{
    std::vector<std::vector<float>> z; // k-faces, one per window
};

static Split EncodeWindows(Encoder& enc, std::span<const float> stream,
                           int count, size_t hop, size_t n, size_t sub,
                           size_t start_window)
{
    Split s;
    s.z.assign(static_cast<size_t>(count), std::vector<float>(sub));
    for (int i = 0; i < count; ++i)
    {
        const size_t off = (start_window + static_cast<size_t>(i)) * hop;
        const float* out = enc.RunEpisode(std::span(stream.data() + off, n));
        for (size_t v = 0; v < sub; ++v)
            s.z[static_cast<size_t>(i)][v] = out[v];
    }
    return s;
}

static float PairMse(Predictor& pred, const Split& s)
{
    if (s.z.size() < 2)
        return 0.f;
    float sum = 0.f;
    const size_t pairs = s.z.size() - 1;
    for (size_t i = 0; i < pairs; ++i)
    {
        const float* hat = pred.Predict(s.z[i]);
        sum += MeanSquare(std::span(hat, s.z[i + 1].size()), s.z[i + 1]);
    }
    return sum / static_cast<float>(pairs);
}

static float IdentityMse(const Split& s)
{
    if (s.z.size() < 2)
        return 0.f;
    float sum = 0.f;
    const size_t pairs = s.z.size() - 1;
    for (size_t i = 0; i < pairs; ++i)
        sum += MeanSquare(s.z[i], s.z[i + 1]);
    return sum / static_cast<float>(pairs);
}

static float NextPower(const Split& s)
{
    if (s.z.size() < 2)
        return 0.f;
    float sum = 0.f;
    const size_t pairs = s.z.size() - 1;
    for (size_t i = 0; i < pairs; ++i)
        sum += MeanSquare(s.z[i + 1]);
    return sum / static_cast<float>(pairs);
}

int main()
{
    auto enc = Encoder::Create(MakeEncoderConfig());
    auto pred = Predictor::Create(MakePredictorConfig());
    const size_t n = enc->Size();
    const size_t sub = size_t{1} << kSubcubeDim;

    if (sub >= n)
        return Fail("k-face is not a compression");
    if (pred->Size() != sub)
        return Fail("Predictor Size");
    if (kHop == 0)
        return Fail("hop is zero");
    if (kTrain < 2 || kVal < 2 || kTest < 2)
        return Fail("each split needs at least two windows");

    const int n_windows = kTrain + kVal + kTest;
    const size_t stream_len =
        static_cast<size_t>(n_windows - 1) * kHop + n;

    PrintEncoderBanner("JepaPredictorTest", *enc, kSubcubeDim, sub);
    {
        const PredictorConfig p = pred->Config();
        const LCNTrainingConfig& t = p.training;
        std::printf("JepaPredictorTest: pred dim=%zu N=%zu z_max=%zu span=%zu "
                    "tanh_last=%d seed=%llu  lr=%.6g lr_min_frac=%.6g "
                    "restore_best=%d\n",
                    p.dim, pred->Size(), p.z_max, p.gather_span,
                    p.tanh_last ? 1 : 0,
                    static_cast<unsigned long long>(p.seed),
                    static_cast<double>(t.lr),
                    static_cast<double>(t.lr_min_frac),
                    t.restore_best ? 1 : 0);
        std::printf("JepaPredictorTest: task train=%d val=%d test=%d epochs=%d "
                    "hop=%zu stream=%zu pairs=%d/%d/%d\n",
                    kTrain, kVal, kTest, kEpochs, kHop, stream_len,
                    kTrain - 1, kVal - 1, kTest - 1);
        std::fflush(stdout);
    }
    PrintSineBanner("JepaPredictorTest", kSineTerms, kCyclesMin, kCyclesMax,
                    kAmpMin, kAmpMax, kDataSeed);

    std::mt19937_64 rng(kDataSeed);
    SineTerm terms[kSineTerms];
    DrawSines(terms, rng);
    for (int t = 0; t < kSineTerms; ++t)
        std::printf("JepaPredictorTest: sine[%d] cycles=%.6g phase=%.6g amp=%.6g\n",
                    t, static_cast<double>(terms[t].cycles),
                    static_cast<double>(terms[t].phase),
                    static_cast<double>(terms[t].amp));
    std::fflush(stdout);

    std::vector<float> stream(stream_len);
    FillSineStream(stream, terms, n);

    Split train = EncodeWindows(*enc, stream, kTrain, kHop, n, sub, 0);
    Split val = EncodeWindows(*enc, stream, kVal, kHop, n, sub,
                              static_cast<size_t>(kTrain));
    Split test = EncodeWindows(*enc, stream, kTest, kHop, n, sub,
                               static_cast<size_t>(kTrain + kVal));

    {
        const float* again = enc->RunEpisode(std::span(stream.data(), n));
        for (size_t i = 0; i < sub; ++i)
            if (again[i] != train.z[0][i])
                return Fail("RunEpisode not repeatable");
    }

    {
        const float* z = enc->RunEpisode(std::span(stream.data(), n));
        std::printf("JepaPredictorTest: stage mean|value| after first train window\n");
        std::printf("JepaPredictorTest:   Field (window on the cube)             "
                    "mean|x|=%.4g  (N=%zu)\n",
                    MeanAbs(std::span(stream.data(), n)), n);
        std::printf("JepaPredictorTest:   Encoder output (full cube, age 0)      "
                    "mean|z|=%.4g  (N=%zu)\n",
                    MeanAbs(std::span(z, n)), n);
        std::printf("JepaPredictorTest:   Encoder k-face (Predictor in/out)      "
                    "mean|s|=%.4g  (sub=%zu)\n",
                    MeanAbs(train.z[0]), sub);
        std::fflush(stdout);
    }

    const float ident = IdentityMse(test);
    const float power = NextPower(test);
    const float first_mse = PairMse(*pred, train);

    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        pred->SetEpoch(epoch, kEpochs);
        pred->BeginBatch();
        for (size_t i = 0; i + 1 < train.z.size(); ++i)
            pred->Accumulate(train.z[i], train.z[i + 1]);
        pred->EndBatch();
        pred->Observe(PairMse(*pred, val), epoch);
    }
    pred->RestoreBest();

    const float train_mse = PairMse(*pred, train);
    const float val_mse = PairMse(*pred, val);
    const float test_mse = PairMse(*pred, test);

    std::printf("JepaPredictorTest: MSE train %.5f -> %.5f  val %.5f  test %.5f  "
                "(identity %.5f  next-power %.5f)\n",
                first_mse, train_mse, val_mse, test_mse, ident, power);
    if (ident > 0.f)
        std::printf("JepaPredictorTest: test / identity = %.4g   test / power = %.4g\n",
                    static_cast<double>(test_mse / ident),
                    power > 0.f ? static_cast<double>(test_mse / power) : 0.0);
    std::fflush(stdout);

    if (!(train_mse < first_mse))
        return Fail("train MSE did not fall");
    if (!(test_mse < power))
        return Fail("test MSE not below next-window power (zero predictor)");

    std::printf("ok\n");
    return 0;
}
