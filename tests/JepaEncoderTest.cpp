// k-face as a JEPA candidate: does the Encoder amplify noise vs content?
//
// For each pair of two-sine fields A and B: encode A, A+noise, and B.
// RMSE in the field vs RMSE on the k-face. The score is
//   (noise/content)_k-face  /  (noise/content)_field
// <1 the k-face damps noise relative to content; ~1 it copies the field;
// >1 it amplifies noise vs content. Reconstruction cannot tell these apart.
// Decoder is not used.

#include "Encoder.h"
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
static constexpr size_t kSubcubeDim = 5; // k < dim; the JEPA candidate face

// =============================================================================
// Encoder configuration — primary knobs (edit here)
// =============================================================================

static EncoderConfig MakeEncoderConfig()
{
    EncoderConfig cfg;
    cfg.dim = kDim;
    cfg.seed = 1; // weight draw
    cfg.spectral_radius = 0.999f;
    cfg.leak_rate = 0.25;
    cfg.input_scaling = 0.8f;
    cfg.history_depth = 8; // M
    cfg.passes = 2 * kDim; // T; 0 → T = N
    cfg.ic_seed = 2; // start state s0
    return cfg;
}

// =============================================================================
// Task parameters (not part of EncoderConfig)
// =============================================================================

static constexpr int kPairs = 1024; // (A, B) sine pairs
static constexpr float kNoiseSigma = 0.2f; // i.i.d. N(0,σ) on A
static constexpr uint64_t kDataSeed = 4; // sine-field generator
static constexpr uint64_t kNoiseSeed = 5; // noise draws

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

static float Rmse(std::span<const float> a, std::span<const float> b)
{
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / static_cast<float>(a.size()));
}

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

static void TakeKFace(Encoder& enc, std::span<const float> field,
                      std::span<float> kface)
{
    const float* z = enc.RunEpisode(field);
    for (size_t i = 0; i < kface.size(); ++i)
        kface[i] = z[i];
}

int main()
{
    auto enc = Encoder::Create(MakeEncoderConfig());
    const size_t n = enc->Size();
    const size_t sub = size_t{1} << kSubcubeDim;
    if (sub >= n)
        return Fail("k-face is not a compression");

    PrintEncoderBanner("JepaEncoderTest", *enc, kSubcubeDim, sub);
    std::printf("JepaEncoderTest: task pairs=%d sigma=%.6g noise_seed=%llu\n",
                kPairs, static_cast<double>(kNoiseSigma),
                static_cast<unsigned long long>(kNoiseSeed));
    std::fflush(stdout);
    PrintSineBanner("JepaEncoderTest", kSineTerms, kCyclesMin, kCyclesMax,
                    kAmpMin, kAmpMax, kDataSeed);

    std::mt19937_64 rng(kDataSeed);
    std::mt19937_64 nrng(kNoiseSeed);
    std::normal_distribution<float> gauss(0.f, kNoiseSigma);

    std::vector<float> a(n), b(n), noisy(n);
    std::vector<float> za(sub), zb(sub), zn(sub);

    {
        FillSineField(a, rng);
        TakeKFace(*enc, a, za);
        const float* again = enc->RunEpisode(a);
        for (size_t i = 0; i < sub; ++i)
            if (again[i] != za[i])
                return Fail("RunEpisode not repeatable");
    }
    rng.seed(kDataSeed);

    double sum_field_noise = 0.0, sum_field_content = 0.0;
    double sum_k_noise = 0.0, sum_k_content = 0.0;

    for (int p = 0; p < kPairs; ++p)
    {
        FillSineField(a, rng);
        FillSineField(b, rng);
        for (size_t i = 0; i < n; ++i)
            noisy[i] = a[i] + gauss(nrng);

        TakeKFace(*enc, a, za);
        TakeKFace(*enc, noisy, zn);
        TakeKFace(*enc, b, zb);

        sum_field_noise += static_cast<double>(Rmse(a, noisy));
        sum_field_content += static_cast<double>(Rmse(a, b));
        sum_k_noise += static_cast<double>(Rmse(za, zn));
        sum_k_content += static_cast<double>(Rmse(za, zb));
    }

    const double np = static_cast<double>(kPairs);
    const double field_noise = sum_field_noise / np;
    const double field_content = sum_field_content / np;
    const double k_noise = sum_k_noise / np;
    const double k_content = sum_k_content / np;
    const double r_field = field_noise / field_content;
    const double r_k = k_noise / k_content;
    const double rel = r_k / r_field;

    std::printf("JepaEncoderTest: field  RMSE noise=%.4g  content=%.4g  "
                "noise/content=%.4g\n",
                field_noise, field_content, r_field);
    std::printf("JepaEncoderTest: k-face RMSE noise=%.4g  content=%.4g  "
                "noise/content=%.4g\n",
                k_noise, k_content, r_k);
    std::printf("JepaEncoderTest: (noise/content)_k-face / "
                "(noise/content)_field = %.4g\n",
                rel);
    std::printf("JepaEncoderTest:   <1 damps noise vs content; ~1 copies the "
        "field; >1 amplifies noise vs content\n");
    std::fflush(stdout);

    if (!(field_content > 1e-3))
        return Fail("field content RMSE collapsed");
    if (!(k_content > 1e-3))
        return Fail("k-face content RMSE collapsed");
    if (!(rel < 2.0))
        return Fail("k-face amplifies noise vs content relative to the field");

    std::printf("ok\n");
    return 0;
}
