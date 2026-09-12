// Encode → Decode: does the k-face still carry the field?

#include "Encoder.h"
#include "Decoder.h"
#include "report_config.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <span>
#include <vector>

// =============================================================================
// Shared cube (Encoder.dim == Decoder.dim)crea
// =============================================================================

static constexpr size_t kDim = 8; // N = 2^dim
static constexpr size_t kSubcubeDim = 5; // k < dim; compression is this cut

// =============================================================================
// Encoder configuration — primary knobs (edit here)
// =============================================================================

static EncoderConfig MakeEncoderConfig()
{
    EncoderConfig cfg;
    cfg.dim = kDim;
    cfg.seed = 1; // weight draw
    cfg.spectral_radius = 0.999f;
    cfg.leak_rate = 0.25f;
    cfg.input_scaling = 0.8f;
    cfg.history_depth = 8; // M
    cfg.passes = 2*kDim; // T; 0 → T = N
    cfg.ic_seed = 2; // start state s0
    return cfg;
}

// =============================================================================
// Decoder configuration — primary knobs (edit here)
// =============================================================================

static DecoderConfig MakeDecoderConfig()
{
    DecoderConfig cfg;
    cfg.dim = kDim;
    cfg.k = kSubcubeDim;
    cfg.z_max = 3 * kDim; // 0 → dim; dim = antipodal reach
    cfg.gather_span = 5;
    cfg.tanh_last = true;
    cfg.seed = 3; // LCN weight draw

    cfg.training.lr = 0.03f;;
    cfg.training.lr_min_frac = 0.05f;
    cfg.training.lr_decay_epochs = 0; // 0 → kEpochs
    cfg.training.restore_best = true;
    cfg.training.beta1 = 0.9f;
    cfg.training.beta2 = 0.999f;
    cfg.training.eps = 1e-8f;
    return cfg;
}

// =============================================================================
// Task parameters (not part of EncoderConfig / DecoderConfig)
// =============================================================================

static constexpr int kTrain = 1024; // training fields
static constexpr int kVal = kTrain / 2; // validation fields (restore_best)
static constexpr int kTest = kTrain / 2; // held-out fields
static constexpr int kEpochs = 2*400; // decoder training epochs
static constexpr uint64_t kDataSeed = 4; // sine-field generator

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

// One field: sines on the cube, vertex i = sample i.
static void FillSineField(std::span<float> x, std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> cycles(kCyclesMin, kCyclesMax);
    std::uniform_real_distribution<float> phase(0.f, 2.f * std::numbers::pi_v<float>);
    std::uniform_real_distribution<float> amp(kAmpMin, kAmpMax);
    const float n = static_cast<float>(x.size());
    for (float& v : x)
        v = 0.f;
    for (int t = 0; t < kSineTerms; ++t)
    {
        const float c = cycles(rng);
        const float p = phase(rng);
        const float a = amp(rng);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] += a * std::sin(2.f * std::numbers::pi_v<float> * c
                * static_cast<float>(i) / n + p);
    }
}

struct Split
{
    std::vector<std::vector<float>> fields;
    std::vector<std::vector<float>> subs;
};

static Split MakeSplit(Encoder& enc, int count, size_t sub, std::mt19937_64& rng)
{
    const size_t n = enc.Size();
    Split s;
    s.fields.assign(static_cast<size_t>(count), std::vector<float>(n));
    s.subs.assign(static_cast<size_t>(count), std::vector<float>(sub));
    for (int i = 0; i < count; ++i)
    {
        FillSineField(s.fields[static_cast<size_t>(i)], rng);
        const float* out = enc.RunEpisode(s.fields[static_cast<size_t>(i)]);
        for (size_t v = 0; v < sub; ++v)
            s.subs[static_cast<size_t>(i)][v] = out[v];
    }
    return s;
}

static float SplitMse(Decoder& dec, const Split& s)
{
    float sum = 0.f;
    for (size_t i = 0; i < s.fields.size(); ++i)
    {
        const float* y = dec.Decode(s.subs[i]);
        sum += MeanSquare(std::span(y, s.fields[i].size()), s.fields[i]);
    }
    return sum / static_cast<float>(s.fields.size());
}

int main()
{
    auto enc = Encoder::Create(MakeEncoderConfig());
    auto dec = Decoder::Create(MakeDecoderConfig());
    const size_t n = enc->Size();
    const size_t sub = dec->CodeSize();

    if (enc->Size() != dec->FieldSize())
        return Fail("Encoder Size and Decoder FieldSize differ");
    if (sub != (size_t{1} << kSubcubeDim))
        return Fail("Decoder CodeSize");

    PrintEncoderBanner("CompressionTest", *enc, kSubcubeDim, sub);
    {
        const DecoderConfig d = dec->Config();
        const LCNTrainingConfig& t = d.training;
        std::printf("CompressionTest: dec  z_max=%zu span=%zu tanh_last=%d "
                    "seed=%llu  lr=%.6g lr_min_frac=%.6g restore_best=%d\n",
                    d.z_max, d.gather_span, d.tanh_last ? 1 : 0,
                    static_cast<unsigned long long>(d.seed),
                    static_cast<double>(t.lr),
                    static_cast<double>(t.lr_min_frac),
                    t.restore_best ? 1 : 0);
        std::printf("CompressionTest: task train=%d val=%d test=%d epochs=%d\n",
                    kTrain, kVal, kTest, kEpochs);
        std::fflush(stdout);
    }
    PrintSineBanner("CompressionTest", kSineTerms, kCyclesMin, kCyclesMax,
                    kAmpMin, kAmpMax, kDataSeed);

    std::mt19937_64 rng(kDataSeed);
    Split train = MakeSplit(*enc, kTrain, sub, rng);
    Split val = MakeSplit(*enc, kVal, sub, rng);
    Split test = MakeSplit(*enc, kTest, sub, rng);

    {
        const float* again = enc->RunEpisode(train.fields[0]);
        for (size_t i = 0; i < sub; ++i)
            if (again[i] != train.subs[0][i])
                return Fail("RunEpisode not repeatable");
    }

    std::vector<float> train_subs_flat;
    train_subs_flat.reserve(train.subs.size() * sub);
    for (const auto& s : train.subs)
        train_subs_flat.insert(train_subs_flat.end(), s.begin(), s.end());
    dec->FitInputScale(train_subs_flat);

    {
        enc->RunEpisode(train.fields[0]);
        const float scale = dec->InputScale();
        std::vector<float> scaled(sub);
        for (size_t i = 0; i < sub; ++i)
            scaled[i] = train.subs[0][i] * scale;
        std::printf("CompressionTest: stage scales after one train field "
            "(mean |value|; ~1 is a live field, ~0 is crushed)\n");
        std::printf("CompressionTest:   Field (sine on the cube)                 "
                    "mean|x|=%.4g  (N=%zu)\n",
                    static_cast<double>(MeanAbs(train.fields[0])), n);
        std::printf("CompressionTest:   Encoder output (full cube, age 0)        "
                    "mean|z|=%.4g  (N=%zu)\n",
                    static_cast<double>(MeanAbs(std::span(enc->RawCube(), n))), n);
        std::printf("CompressionTest:   Encoder k-face (compression)             "
                    "mean|s|=%.4g  (sub=%zu)\n",
                    static_cast<double>(MeanAbs(train.subs[0])), sub);
        std::printf("CompressionTest:   Decoder input (k-face * input_scale)     "
                    "mean|f|=%.4g  (sub=%zu, scale=%.4g)\n",
                    static_cast<double>(MeanAbs(scaled)), sub, static_cast<double>(scale));
        std::fflush(stdout);
    }

    const float first_mse = SplitMse(*dec, train);
    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        dec->SetEpoch(epoch, kEpochs);
        dec->BeginBatch();
        for (size_t i = 0; i < train.fields.size(); ++i)
            dec->Accumulate(train.subs[i], train.fields[i]);
        dec->EndBatch();
        dec->Observe(SplitMse(*dec, val), epoch);
    }
    dec->RestoreBest();

    const float train_mse = SplitMse(*dec, train);
    const float val_mse = SplitMse(*dec, val);
    const float test_mse = SplitMse(*dec, test);

    float test_power = 0.f;
    for (const auto& f : test.fields)
        test_power += MeanSquare(f);
    test_power /= static_cast<float>(test.fields.size());

    std::printf("CompressionTest: dec  input_scale=%.6g\n",
                static_cast<double>(dec->InputScale()));
    std::printf("CompressionTest: MSE train %.5f -> %.5f  val %.5f  test %.5f  "
                "(field power %.5f)\n",
                first_mse, train_mse, val_mse, test_mse, test_power);

    if (!(train_mse < first_mse))
        return Fail("train MSE did not fall");
    if (!(test_mse < test_power))
        return Fail("test MSE not below field power (zero predictor)");

    std::printf("ok\n");
    return 0;
}