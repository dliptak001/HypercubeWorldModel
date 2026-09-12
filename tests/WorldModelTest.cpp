// Many two-sine draws through WorldModel: train on a mix, score held-out draws.
// Pairs are consecutive windows inside a song. Decoder is not used.

#include "WorldModel.h"
#include "ThreadPool.h"
#include "report_config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <thread>
#include <vector>

// =============================================================================
// Shared cube
// =============================================================================

static constexpr size_t kDim = 8; // N = 2^dim
static constexpr size_t kSubcubeDim = 5; // k < dim; WorldModel latent face

// =============================================================================
// WorldModel configuration — primary knobs (edit here)
// =============================================================================

static WorldModelConfig MakeWorldModelConfig()
{
    WorldModelConfig cfg;
    cfg.encoder.dim = kDim;
    cfg.encoder.seed = 1; // weight draw
    cfg.encoder.spectral_radius = 0.999f;
    cfg.encoder.leak_rate = 0.25f;
    cfg.encoder.input_scaling = 0.8f;
    cfg.encoder.history_depth = 8; // M
    cfg.encoder.passes = 2 * kDim; // T; 0 → T = N
    cfg.encoder.ic_seed = 2; // start state s0

    cfg.k = kSubcubeDim;
    cfg.predictor.z_max = 3 * kSubcubeDim; // 0 → k+1; k = antipodal reach
    cfg.predictor.gather_span = 5;;
    cfg.predictor.tanh_last = true;
    cfg.predictor.seed = 3; // Predictor LCN weight draw

    cfg.predictor.training.lr = 0.03f;
    cfg.predictor.training.lr_min_frac = 0.05f;
    cfg.predictor.training.lr_decay_epochs = 0; // 0 → kEpochs
    cfg.predictor.training.restore_best = true;
    cfg.predictor.training.beta1 = 0.9f;
    cfg.predictor.training.beta2 = 0.999f;
    cfg.predictor.training.eps = 1e-8f;
    return cfg;
}

// =============================================================================
// Task parameters (not part of WorldModelConfig)
// =============================================================================

static constexpr int kTrainSongs = 640; // disjoint two-sine draws
static constexpr int kValSongs = 128; // disjoint from train; restore_best
static constexpr int kTestSongs = 128; // disjoint from train and val
static constexpr int kWindows = 8; // consecutive windows per song (pairs = kWindows - 1)
static constexpr int kEpochs = 800;
static constexpr int kWorkers = 0; // 0 → hardware_concurrency
static constexpr size_t kHop = size_t{1} << kDim;
static constexpr uint64_t kDataSeed = 4;

// Two-sine has no motor. Dummy action field of length 2^k, through the
// action encoder once.
static constexpr float kActionFill = 1.f;

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

static double ElapsedSec(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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

using Codes = std::vector<std::vector<float>>; // windows of one song

static Codes EncodeSong(WorldModel& wm, std::span<const float> stream,
                        int n_windows, size_t hop, size_t n)
{
    Codes z(static_cast<size_t>(n_windows), std::vector<float>(wm.CodeSize()));
    for (int i = 0; i < n_windows; ++i)
    {
        const size_t off = static_cast<size_t>(i) * hop;
        wm.Encode(std::span(stream.data() + off, n), z[static_cast<size_t>(i)]);
    }
    return z;
}

struct Pool
{
    std::vector<Codes> songs;
    std::vector<float> first_field; // first window of song 0, length N
};

static Pool MakePool(WorldModel& wm, int n_songs,
                     std::mt19937_64& rng, size_t hop, size_t n)
{
    Pool p;
    p.songs.resize(static_cast<size_t>(n_songs));
    const size_t len = static_cast<size_t>(kWindows - 1) * hop + n;
    std::vector<float> stream(len);
    SineTerm terms[kSineTerms];
    for (int s = 0; s < n_songs; ++s)
    {
        DrawSines(terms, rng);
        FillSineStream(stream, terms, n);
        p.songs[static_cast<size_t>(s)] = EncodeSong(wm, stream, kWindows, hop, n);
        if (s == 0)
            p.first_field.assign(stream.data(), stream.data() + n);
    }
    return p;
}

static int PairCount(const Pool& p)
{
    int n = 0;
    for (const auto& z : p.songs)
        if (z.size() >= 2)
            n += static_cast<int>(z.size() - 1);
    return n;
}

static float IdentityMse(const Pool& p)
{
    float sum = 0.f;
    int pairs = 0;
    for (const auto& z : p.songs)
    {
        if (z.size() < 2)
            continue;
        for (size_t i = 0; i + 1 < z.size(); ++i)
        {
            sum += MeanSquare(z[i], z[i + 1]);
            ++pairs;
        }
    }
    return pairs > 0 ? sum / static_cast<float>(pairs) : 0.f;
}

static float NextPower(const Pool& p)
{
    float sum = 0.f;
    int pairs = 0;
    for (const auto& z : p.songs)
    {
        if (z.size() < 2)
            continue;
        for (size_t i = 0; i + 1 < z.size(); ++i)
        {
            sum += MeanSquare(z[i + 1]);
            ++pairs;
        }
    }
    return pairs > 0 ? sum / static_cast<float>(pairs) : 0.f;
}

static void PrintScore(const char* tag, float mse, float ident, float power)
{
    const double r_id = ident > 0.f ? static_cast<double>(mse / ident) : 0.0;
    const double r_pw = power > 0.f ? static_cast<double>(mse / power) : 0.0;
    std::printf("%-5s mse=%.5f  identity=%.5f  next-power=%.5f  "
                "mse/ident=%.4g  mse/power=%.4g\n",
                tag, static_cast<double>(mse),
                static_cast<double>(ident), static_cast<double>(power),
                r_id, r_pw);
    std::fflush(stdout);
}

static size_t ResolveWorkers(size_t n_songs)
{
    size_t w = 0;
    if (kWorkers > 0)
        w = static_cast<size_t>(kWorkers);
    else
    {
        const unsigned hw = std::thread::hardware_concurrency();
        w = hw ? static_cast<size_t>(hw) : 1;
    }
    if (w < 1)
        w = 1;
    if (w > n_songs)
        w = n_songs;
    return w;
}

static PredictorConfig MakeReplicaConfig(const WorldModelConfig& c)
{
    PredictorConfig p;
    p.dim = c.k + 1;
    p.z_max = c.predictor.z_max;
    p.gather_span = c.predictor.gather_span;
    p.tanh_last = c.predictor.tanh_last;
    p.seed = c.predictor.seed;
    p.training = c.predictor.training;
    p.training.restore_best = false;
    return p;
}

static void Shard(size_t t, size_t w, size_t n, size_t& lo, size_t& hi)
{
    lo = t * n / w;
    hi = (t + 1) * n / w;
}

static void SyncReplicas(WorldModel& wm,
                         std::vector<std::unique_ptr<Predictor>>& reps)
{
    const auto& weights = wm.Weights();
    for (auto& p : reps)
        p->LoadWeights(weights);
}

static float ParallelPairMse(ThreadPool& pool, WorldModel& wm,
                             std::vector<std::unique_ptr<Predictor>>& reps,
                             std::span<const float> za, const Pool& data)
{
    const size_t w = reps.size();
    const size_t n = data.songs.size();
    if (n == 0 || w == 0)
        return 0.f;
    std::vector<double> sums(w, 0.0);
    std::vector<int> counts(w, 0);
    pool.Run([&](size_t t)
    {
        size_t lo = 0, hi = 0;
        Shard(t, w, n, lo, hi);
        double s = 0.0;
        int c = 0;
        Predictor& pred = *reps[t];
        std::vector<float> packed(pred.Size());
        for (size_t si = lo; si < hi; ++si)
        {
            const auto& z = data.songs[si];
            for (size_t i = 0; i + 1 < z.size(); ++i)
            {
                wm.Pack(z[i], za, packed);
                const float* hat = pred.Predict(packed);
                s += static_cast<double>(
                    MeanSquare(std::span(hat, z[i + 1].size()), z[i + 1]));
                ++c;
            }
        }
        sums[t] = s;
        counts[t] = c;
    });
    double s = 0.0;
    int c = 0;
    for (size_t t = 0; t < w; ++t)
    {
        s += sums[t];
        c += counts[t];
    }
    return c > 0 ? static_cast<float>(s / static_cast<double>(c)) : 0.f;
}

static void ParallelTrainEpoch(ThreadPool& pool, WorldModel& wm,
                               std::vector<std::unique_ptr<Predictor>>& reps,
                               std::span<const float> za, const Pool& train)
{
    const size_t w = reps.size();
    const size_t n = train.songs.size();
    const auto& weights = wm.Weights();
    pool.Run([&](size_t t)
    {
        Predictor& pred = *reps[t];
        pred.LoadWeights(weights);
        pred.BeginBatch();
        size_t lo = 0, hi = 0;
        Shard(t, w, n, lo, hi);
        std::vector<float> packed(pred.Size());
        for (size_t si = lo; si < hi; ++si)
        {
            const auto& z = train.songs[si];
            for (size_t i = 0; i + 1 < z.size(); ++i)
            {
                wm.Pack(z[i], za, packed);
                pred.Accumulate(packed, z[i + 1]);
            }
        }
    });
    wm.BeginBatch();
    for (auto& p : reps)
        wm.AddGrad(p->Grad());
    wm.EndBatch();
}

int main()
{
    auto wm = WorldModel::Create(MakeWorldModelConfig());
    const size_t n = wm->FieldSize();
    const size_t sub = wm->CodeSize();

    if (sub >= n)
        return Fail("k-face is not a compression");
    if (kHop == 0)
        return Fail("hop is zero");
    if (kWindows < 2)
        return Fail("each song needs at least two windows");
    if (kTrainSongs < 1 || kValSongs < 1 || kTestSongs < 1)
        return Fail("each pool needs at least one song");

    std::vector<float> a_field(sub, kActionFill); // action cube is the k-face size
    std::vector<float> za(sub);
    wm->EncodeAction(a_field, za);

    const int pairs_train = kTrainSongs * (kWindows - 1);
    const int pairs_val = kValSongs * (kWindows - 1);
    const int pairs_test = kTestSongs * (kWindows - 1);
    const size_t workers = ResolveWorkers(static_cast<size_t>(kTrainSongs));

    {
        const WorldModelConfig c = wm->Config();
        PrintEncoderBanner("", c.encoder, n,
                           wm->RealizedSpectralRadius(), c.k, sub);
        const LCNTrainingConfig& t = c.predictor.training;
        std::printf("pred z_max=%zu span=%zu tanh_last=%d seed=%llu\n",
                    c.predictor.z_max, c.predictor.gather_span, c.predictor.tanh_last ? 1 : 0,
                    static_cast<unsigned long long>(c.predictor.seed));
        std::printf("adam lr=%.6g lr_min_frac=%.6g lr_decay_epochs=%d "
                    "restore_best=%d beta1=%.6g beta2=%.6g eps=%.6g\n",
                    static_cast<double>(t.lr),
                    static_cast<double>(t.lr_min_frac),
                    t.lr_decay_epochs, t.restore_best ? 1 : 0,
                    static_cast<double>(t.beta1),
                    static_cast<double>(t.beta2),
                    static_cast<double>(t.eps));
        std::printf("task songs=%d/%d/%d windows=%d pairs=%d/%d/%d "
                    "epochs=%d hop=%zu workers=%zu\n",
                    kTrainSongs, kValSongs, kTestSongs, kWindows,
                    pairs_train, pairs_val, pairs_test,
                    kEpochs, kHop, workers);
        std::fflush(stdout);
    }
    PrintSineBanner("", kSineTerms, kCyclesMin, kCyclesMax,
                    kAmpMin, kAmpMax, kDataSeed);

    std::mt19937_64 rng(kDataSeed);
    const auto t_dataset = std::chrono::steady_clock::now();
    Pool train = MakePool(*wm, kTrainSongs, rng, kHop, n);
    Pool val = MakePool(*wm, kValSongs, rng, kHop, n);
    Pool test = MakePool(*wm, kTestSongs, rng, kHop, n);
    const double sec_dataset = ElapsedSec(t_dataset);

    if (PairCount(train) != pairs_train || PairCount(val) != pairs_val
        || PairCount(test) != pairs_test)
        return Fail("pair counts");

    {
        std::vector<float> again(sub);
        wm->Encode(train.first_field, again);
        for (size_t i = 0; i < sub; ++i)
            if (again[i] != train.songs[0][0][i])
                return Fail("Encode not repeatable");
    }

    {
        std::printf("stage mean|x|=%.4g  mean|s_train|=%.4g  "
                    "mean|s_val|=%.4g  mean|s_test|=%.4g\n",
                    static_cast<double>(MeanAbs(train.first_field)),
                    static_cast<double>(MeanAbs(train.songs[0][0])),
                    static_cast<double>(MeanAbs(val.songs[0][0])),
                    static_cast<double>(MeanAbs(test.songs[0][0])));
        std::fflush(stdout);
    }

    const float ident_train = IdentityMse(train);
    const float power_train = NextPower(train);
    const float ident_val = IdentityMse(val);
    const float power_val = NextPower(val);
    const float ident_test = IdentityMse(test);
    const float power_test = NextPower(test);

    ThreadPool pool(workers);
    std::vector<std::unique_ptr<Predictor>> reps;
    reps.reserve(workers);
    {
        const PredictorConfig pcfg = MakeReplicaConfig(wm->Config());
        for (size_t t = 0; t < workers; ++t)
            reps.push_back(Predictor::Create(pcfg));
    }

    SyncReplicas(*wm, reps);
    const auto t_eval0 = std::chrono::steady_clock::now();
    const float first_mse = ParallelPairMse(pool, *wm, reps, za, train);
    const double sec_eval_pre = ElapsedSec(t_eval0);

    const auto t_train = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        wm->SetEpoch(epoch, kEpochs);
        ParallelTrainEpoch(pool, *wm, reps, za, train);
        SyncReplicas(*wm, reps);
        wm->Observe(ParallelPairMse(pool, *wm, reps, za, val), epoch);
    }
    const double sec_train = ElapsedSec(t_train);

    const auto t_eval1 = std::chrono::steady_clock::now();
    wm->RestoreBest();
    SyncReplicas(*wm, reps);
    const float train_mse = ParallelPairMse(pool, *wm, reps, za, train);
    const float val_mse = ParallelPairMse(pool, *wm, reps, za, val);
    const float test_mse = ParallelPairMse(pool, *wm, reps, za, test);
    const double sec_eval = sec_eval_pre + ElapsedSec(t_eval1);

    std::printf("train mse=%.5f -> %.5f  identity=%.5f  next-power=%.5f  "
                "mse/ident=%.4g  mse/power=%.4g\n",
                static_cast<double>(first_mse), static_cast<double>(train_mse),
                static_cast<double>(ident_train), static_cast<double>(power_train),
                ident_train > 0.f ? static_cast<double>(train_mse / ident_train) : 0.0,
                power_train > 0.f ? static_cast<double>(train_mse / power_train) : 0.0);
    std::fflush(stdout);
    PrintScore("val", val_mse, ident_val, power_val);
    PrintScore("test", test_mse, ident_test, power_test);
    std::printf("time dataset=%.3fs  train=%.3fs  eval=%.3fs  total=%.3fs\n",
                sec_dataset, sec_train, sec_eval,
                sec_dataset + sec_train + sec_eval);
    std::printf("Look at test mse/power: you want it as low as you can get, "
                "with val mse/power about the same number.\n");
    std::fflush(stdout);

    if (!(train_mse < first_mse))
        return Fail("train MSE did not fall");
    if (!(test_mse < power_test))
        return Fail("test MSE not below next-window power (zero predictor)");

    std::printf("ok\n");
    return 0;
}
