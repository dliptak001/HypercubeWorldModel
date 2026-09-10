// Terrain walker through WorldModel: train on a mix of maps, score held maps.
// Pairs are consecutive crops inside one walk. Decoder is not used.

#include "WorldModel.h"
#include "ThreadPool.h"
#include "report_config.h"
#include "Map.h"
#include "Walker.h"
#include "ActionField.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <span>
#include <thread>
#include <vector>

// =============================================================================
// Shared cube
// =============================================================================

static constexpr size_t kDim = 8; // N = 2^dim
static constexpr size_t kSubcubeDim = 6; // k < dim; WorldModel latent face

// Multiplier on E(a) at Pack. 1 = raw action code. The stage-scale lines
// print mean|s| for E(x) and mean|sa| for E(a); set this to their ratio
// to feed P both channels at the same level.
static constexpr float kActionScale = 0.33f;

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
    cfg.action_scale = kActionScale;
    cfg.predictor.z_max = 3 * kSubcubeDim; // 0 → k+1
    cfg.predictor.gather_span = 5;
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
//
// Knob	Now	Try	Why
// kEpochs	800	2400	3× the update count. Same data. Cost is linear: k=7 goes from about 6.5 min to about 20 min.
// kWalkLength	16	32	Doubles pairs per map at no extra map cost. Cheapest data increase if you want one.
// cfg.predictor.z_max	3k	4k	More depth in P. Cost roughly 4/3 per epoch.
// cfg.gather_span	5	6	Upper limit of the LCN range. Small cost.
// kTrainMaps	640	1280	Only after the above, and only if train drops well below val. Doubles dataset and epoch time.

// =============================================================================

static constexpr int kMapSize = 64;
static constexpr int kView = 16; // kView * kView == N
static constexpr float kBarrierFrac = 0.12f;
static constexpr int kWalkLength = 16; // steps; pairs <= this
static constexpr int kTrainMaps = 640;
static constexpr int kValMaps = 128;
static constexpr int kTestMaps = 128;
static constexpr int kEpochs = 2*800;
static constexpr int kWorkers = 0; // 0 → hardware_concurrency
static constexpr uint64_t kDataSeed = 4;

static constexpr int kElevTerms = 3;
static constexpr float kElevCyclesMin = 0.5f;
static constexpr float kElevCyclesMax = 4.f;
static constexpr float kElevAmpMin = 0.3f;
static constexpr float kElevAmpMax = 1.f;

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

static double MeanAbs(std::span<const float> x)
{
    if (x.empty())
        return 0.0;
    double a = 0.0;
    for (float v : x)
        a += std::fabs(static_cast<double>(v));
    return a / static_cast<double>(x.size());
}

using ActionCodes = std::array<std::vector<float>, kHeadingCount>;

struct WalkCodes
{
    std::vector<std::vector<float>> z;
    std::vector<int> action;
    std::vector<char> turned;
    std::vector<char> changed;
};

struct Pool
{
    std::vector<WalkCodes> walks;
    std::vector<float> first_field;
    int pairs = 0;
    int turns = 0;
    int saw_goal = 0;
    int reached = 0;
};

static Walk DrawWalk(std::mt19937_64& rng)
{
    std::uniform_int_distribution<int> hd(0, kHeadingCount - 1);
    for (int t = 0; t < 32; ++t)
    {
        TerrainMap map = TerrainMap::Draw(kMapSize, kView, kBarrierFrac,
                                          kElevTerms, kElevCyclesMin,
                                          kElevCyclesMax, kElevAmpMin,
                                          kElevAmpMax, rng);
        int r = 0, c = 0;
        if (!map.SampleFreePose(rng, r, c))
            continue;
        Walk w = RecordWalk(map, r, c, HeadingFromIndex(hd(rng)),
                            kWalkLength, rng);
        if (!w.actions.empty())
            return w;
    }
    return {};
}

static Pool MakePool(WorldModel& wm, int n_maps, std::mt19937_64& rng)
{
    Pool p;
    p.walks.resize(static_cast<size_t>(n_maps));
    const size_t sub = wm.CodeSize();
    for (int s = 0; s < n_maps; ++s)
    {
        Walk w = DrawWalk(rng);
        if (w.actions.empty())
            return {};
        WalkCodes& wc = p.walks[static_cast<size_t>(s)];
        wc.z.resize(w.frames.size(), std::vector<float>(sub));
        wc.action.resize(w.actions.size());
        wc.turned = w.turned;
        wc.changed = w.changed;
        for (size_t i = 0; i < w.frames.size(); ++i)
            wm.Encode(w.frames[i], wc.z[i]);
        for (size_t i = 0; i < w.actions.size(); ++i)
            wc.action[i] = Index(w.actions[i]);
        p.pairs += static_cast<int>(w.actions.size());
        for (char t : w.turned)
            if (t)
                ++p.turns;
        if (w.saw_goal)
            ++p.saw_goal;
        if (w.reached)
            ++p.reached;
        if (s == 0)
            p.first_field = w.frames[0];
    }
    return p;
}

static float IdentityMse(const Pool& p)
{
    float sum = 0.f;
    int pairs = 0;
    for (const auto& w : p.walks)
    {
        for (size_t i = 0; i + 1 < w.z.size(); ++i)
        {
            sum += MeanSquare(w.z[i], w.z[i + 1]);
            ++pairs;
        }
    }
    return pairs > 0 ? sum / static_cast<float>(pairs) : 0.f;
}

static float NextPower(const Pool& p)
{
    float sum = 0.f;
    int pairs = 0;
    for (const auto& w : p.walks)
    {
        for (size_t i = 0; i + 1 < w.z.size(); ++i)
        {
            sum += MeanSquare(w.z[i + 1]);
            ++pairs;
        }
    }
    return pairs > 0 ? sum / static_cast<float>(pairs) : 0.f;
}

struct Slice
{
    float mse = 0.f;
    float ident = 0.f;
    float power = 0.f;
    int n = 0;
};

// Swap check: the same E(x) through P with every za. If P reads a, the
// true heading scores lower than the wrong ones and is the argmin.
struct Swap
{
    float mse_true = 0.f;  // error with the executed heading (== all.mse)
    float mse_wrong = 0.f; // mean error over the three other headings
    float spread = 0.f;    // mean ||P(z, a) - P(z, a')||^2 over the three a'
    float argmin = 0.f;    // fraction of pairs where the lowest error is the true a
    int n = 0;
};

struct PairScore
{
    Slice all;
    Slice straight; // not an encounter
    Slice turn;     // 90°/180° recover
    Slice keep;     // executed heading == heading before the step
    Slice change;   // executed heading flipped
    Swap swap;
};

struct SwapAcc
{
    double mse_wrong = 0.0;
    double spread = 0.0;
    int hits = 0;
    int n = 0;

    void fold(const SwapAcc& o)
    {
        mse_wrong += o.mse_wrong;
        spread += o.spread;
        hits += o.hits;
        n += o.n;
    }
};

struct SliceAcc
{
    double mse = 0.0;
    double ident = 0.0;
    double power = 0.0;
    int n = 0;

    void add(double e, double id, double pw)
    {
        mse += e;
        ident += id;
        power += pw;
        ++n;
    }

    void fold(const SliceAcc& o)
    {
        mse += o.mse;
        ident += o.ident;
        power += o.power;
        n += o.n;
    }

    Slice mean() const
    {
        Slice s;
        s.n = n;
        if (n > 0)
        {
            const double d = static_cast<double>(n);
            s.mse = static_cast<float>(mse / d);
            s.ident = static_cast<float>(ident / d);
            s.power = static_cast<float>(power / d);
        }
        return s;
    }
};

static void PrintSlice(const char* name, const Slice& s)
{
    const double r_id = s.ident > 0.f ? static_cast<double>(s.mse / s.ident) : 0.0;
    const double r_pw = s.power > 0.f ? static_cast<double>(s.mse / s.power) : 0.0;
    std::printf("      %-8s n=%d mse=%.5f  identity=%.5f  next-power=%.5f  "
                "mse/ident=%.4g  mse/power=%.4g\n",
                name, s.n, static_cast<double>(s.mse),
                static_cast<double>(s.ident), static_cast<double>(s.power),
                r_id, r_pw);
}

static void PrintScore(const char* tag, const PairScore& s)
{
    const Slice& a = s.all;
    const double r_id = a.ident > 0.f ? static_cast<double>(a.mse / a.ident) : 0.0;
    const double r_pw = a.power > 0.f ? static_cast<double>(a.mse / a.power) : 0.0;
    std::printf("%-5s mse=%.5f  identity=%.5f  next-power=%.5f  "
                "mse/ident=%.4g  mse/power=%.4g\n",
                tag, static_cast<double>(a.mse),
                static_cast<double>(a.ident), static_cast<double>(a.power),
                r_id, r_pw);
    PrintSlice("straight", s.straight);
    PrintSlice("turn", s.turn);
    PrintSlice("keep", s.keep);
    PrintSlice("change", s.change);
    const Swap& w = s.swap;
    std::printf("      swap     n=%d mse_true=%.5f  mse_wrong=%.5f  wrong/true=%.4g  "
                "spread=%.5f  argmin==a=%.2f%%\n",
                w.n, static_cast<double>(w.mse_true), static_cast<double>(w.mse_wrong),
                w.mse_true > 0.f ? static_cast<double>(w.mse_wrong / w.mse_true) : 0.0,
                static_cast<double>(w.spread), 100.0 * static_cast<double>(w.argmin));
    std::fflush(stdout);
}

static size_t ResolveWorkers(size_t n_maps)
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
    if (w > n_maps)
        w = n_maps;
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

static PairScore ParallelPairMse(ThreadPool& pool, WorldModel& wm,
                                 std::vector<std::unique_ptr<Predictor>>& reps,
                                 const ActionCodes& za, const Pool& data)
{
    const size_t w = reps.size();
    const size_t n = data.walks.size();
    PairScore out;
    if (n == 0 || w == 0)
        return out;
    std::vector<SliceAcc> acc_all(w), acc_st(w), acc_tu(w), acc_ke(w), acc_ch(w);
    std::vector<SwapAcc> acc_sw(w);
    pool.Run([&](size_t t)
    {
        size_t lo = 0, hi = 0;
        Shard(t, w, n, lo, hi);
        Predictor& pred = *reps[t];
        std::vector<float> packed(pred.Size());
        const size_t sub = wm.CodeSize();
        std::array<std::vector<float>, kHeadingCount> hat;
        for (auto& h : hat)
            h.resize(sub);
        for (size_t si = lo; si < hi; ++si)
        {
            const auto& walk = data.walks[si];
            for (size_t i = 0; i + 1 < walk.z.size(); ++i)
            {
                const int a = walk.action[i];
                // one Forward per heading on the same E(x)
                double err[kHeadingCount];
                int best = 0;
                for (int h = 0; h < kHeadingCount; ++h)
                {
                    wm.Pack(walk.z[i], za[static_cast<size_t>(h)], packed);
                    const float* out = pred.Predict(packed);
                    std::copy(out, out + sub, hat[static_cast<size_t>(h)].begin());
                    err[h] = static_cast<double>(
                        MeanSquare(hat[static_cast<size_t>(h)], walk.z[i + 1]));
                    if (err[h] < err[best])
                        best = h;
                }
                const double e = err[a];
                const double id = static_cast<double>(
                    MeanSquare(walk.z[i], walk.z[i + 1]));
                const double pw = static_cast<double>(MeanSquare(walk.z[i + 1]));
                double wrong = 0.0, spread = 0.0;
                for (int h = 0; h < kHeadingCount; ++h)
                {
                    if (h == a)
                        continue;
                    wrong += err[h];
                    spread += static_cast<double>(
                        MeanSquare(hat[static_cast<size_t>(h)],
                                   hat[static_cast<size_t>(a)]));
                }
                acc_sw[t].mse_wrong += wrong / (kHeadingCount - 1);
                acc_sw[t].spread += spread / (kHeadingCount - 1);
                acc_sw[t].hits += (best == a) ? 1 : 0;
                ++acc_sw[t].n;
                acc_all[t].add(e, id, pw);
                if (walk.turned[i])
                    acc_tu[t].add(e, id, pw);
                else
                    acc_st[t].add(e, id, pw);
                if (walk.changed[i])
                    acc_ch[t].add(e, id, pw);
                else
                    acc_ke[t].add(e, id, pw);
            }
        }
    });
    SliceAcc all, st, tu, ke, ch;
    SwapAcc sw;
    for (size_t t = 0; t < w; ++t)
    {
        all.fold(acc_all[t]);
        st.fold(acc_st[t]);
        tu.fold(acc_tu[t]);
        ke.fold(acc_ke[t]);
        ch.fold(acc_ch[t]);
        sw.fold(acc_sw[t]);
    }
    out.all = all.mean();
    out.straight = st.mean();
    out.turn = tu.mean();
    out.keep = ke.mean();
    out.change = ch.mean();
    out.swap.n = sw.n;
    if (sw.n > 0)
    {
        const double d = static_cast<double>(sw.n);
        out.swap.mse_true = out.all.mse;
        out.swap.mse_wrong = static_cast<float>(sw.mse_wrong / d);
        out.swap.spread = static_cast<float>(sw.spread / d);
        out.swap.argmin = static_cast<float>(sw.hits / d);
    }
    return out;
}

static void ParallelTrainEpoch(ThreadPool& pool, WorldModel& wm,
                               std::vector<std::unique_ptr<Predictor>>& reps,
                               const ActionCodes& za, const Pool& train)
{
    const size_t w = reps.size();
    const size_t n = train.walks.size();
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
            const auto& walk = train.walks[si];
            for (size_t i = 0; i + 1 < walk.z.size(); ++i)
            {
                wm.Pack(walk.z[i], za[static_cast<size_t>(walk.action[i])], packed);
                pred.Accumulate(packed, walk.z[i + 1]);
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
    const size_t view_n = static_cast<size_t>(kView) * static_cast<size_t>(kView);

    if (sub >= n)
        return Fail("k-face is not a compression");
    if (view_n != n)
        return Fail("view*view must equal FieldSize()");
    if (kWalkLength < 1)
        return Fail("walk length");
    if (kTrainMaps < 1 || kValMaps < 1 || kTestMaps < 1)
        return Fail("each pool needs at least one map");

    // action field: a rows × 16 strip of length 2^k, through the k-cube
    // action encoder; its whole output is E(a)
    const int act_cols = kView;
    const int act_rows = static_cast<int>(sub) / act_cols;
    if (act_rows < 2 || static_cast<size_t>(act_rows) * act_cols != sub)
        return Fail("action strip: 2^k must be at least two rows of view columns");

    ActionCodes za;
    {
        std::vector<float> field(sub);
        for (int h = 0; h < kHeadingCount; ++h)
        {
            za[static_cast<size_t>(h)].assign(sub, 0.f);
            ActionField::Paint(HeadingFromIndex(h), act_rows, act_cols, field);
            wm->EncodeAction(field, za[static_cast<size_t>(h)]);
        }
    }

    const size_t workers = ResolveWorkers(static_cast<size_t>(kTrainMaps));

    {
        const WorldModelConfig c = wm->Config();
        PrintEncoderBanner("", c.encoder, n,
                           wm->RealizedSpectralRadius(), c.k, sub);
        const EncoderConfig a = wm->ActionEncoderConfig();
        std::printf("act  dim=%zu N=%zu  SR_post=%.4g T=%zu  strip=%dx%d  "
                    "scale=%.4g (same seeds and knobs as enc)\n",
                    a.dim, sub,
                    static_cast<double>(wm->ActionRealizedSpectralRadius()),
                    a.passes, act_rows, act_cols,
                    static_cast<double>(c.action_scale));
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
        std::printf("task maps=%d/%d/%d map=%d view=%d L=%d barrier=%.3g "
                    "elev=sines(%d) epochs=%d workers=%zu\n",
                    kTrainMaps, kValMaps, kTestMaps, kMapSize, kView,
                    kWalkLength, static_cast<double>(kBarrierFrac),
                    kElevTerms, kEpochs, workers);
        std::fflush(stdout);
    }

    std::mt19937_64 rng(kDataSeed);
    const auto t_dataset = std::chrono::steady_clock::now();
    Pool train = MakePool(*wm, kTrainMaps, rng);
    Pool val = MakePool(*wm, kValMaps, rng);
    Pool test = MakePool(*wm, kTestMaps, rng);
    const double sec_dataset = ElapsedSec(t_dataset);

    if (train.walks.size() != static_cast<size_t>(kTrainMaps) || train.pairs < 1)
        return Fail("train walks");
    if (val.walks.size() != static_cast<size_t>(kValMaps) || val.pairs < 1)
        return Fail("val walks");
    if (test.walks.size() != static_cast<size_t>(kTestMaps) || test.pairs < 1)
        return Fail("test walks");

    {
        std::vector<float> again(sub);
        wm->Encode(train.first_field, again);
        for (size_t i = 0; i < sub; ++i)
            if (again[i] != train.walks[0].z[0][i])
                return Fail("Encode not repeatable");
    }

    {
        std::vector<float> z_x(wm->LastCube(), wm->LastCube() + n);
        const int h0 = train.walks[0].action[0];
        std::vector<float> a_field(sub);
        ActionField::Paint(HeadingFromIndex(h0), act_rows, act_cols, a_field);
        std::vector<float> packed(2 * sub);
        wm->Pack(train.walks[0].z[0], za[static_cast<size_t>(h0)], packed);

        std::printf("TerrainWalkerTest: stage scales after one train field "
                    "(mean |value|; ~1 is a live field, ~0 is crushed)\n");
        std::printf("TerrainWalkerTest:   View (16x16 crop)                      "
                    "mean|x|=%.4g  (N=%zu)\n",
                    MeanAbs(train.first_field), n);
        std::printf("TerrainWalkerTest:   Encoder output (view, full cube)       "
                    "mean|z|=%.4g  (N=%zu)\n",
                    MeanAbs(z_x), n);
        std::printf("TerrainWalkerTest:   E(x) k-face                            "
                    "mean|s|=%.4g  (sub=%zu)\n",
                    MeanAbs(train.walks[0].z[0]), sub);
        std::printf("TerrainWalkerTest:   Action field (%dx%d half-plane)         "
                    "mean|a|=%.4g  (sub=%zu)\n",
                    act_rows, act_cols, MeanAbs(a_field), sub);
        std::printf("TerrainWalkerTest:   E(a) (action k-cube, whole output)     "
                    "mean|sa|=%.4g  (sub=%zu)\n",
                    MeanAbs(za[static_cast<size_t>(h0)]), sub);
        std::printf("TerrainWalkerTest:   P input (E(x) cat scale*E(a))          "
                    "mean|p|=%.4g  E(x) half=%.4g  E(a) half=%.4g  (2*sub=%zu)\n",
                    MeanAbs(packed),
                    MeanAbs(std::span(packed.data(), sub)),
                    MeanAbs(std::span(packed.data() + sub, sub)),
                    packed.size());
        std::printf("TerrainWalkerTest:   E(a) N/E/S/W                           "
                    "mean|sa|=%.4g %.4g %.4g %.4g  (sub=%zu)\n",
                    MeanAbs(za[0]), MeanAbs(za[1]), MeanAbs(za[2]), MeanAbs(za[3]),
                    sub);
        std::printf("walks train pairs=%d turns=%d saw_goal=%d reached=%d  "
                    "val pairs=%d  test pairs=%d\n",
                    train.pairs, train.turns, train.saw_goal, train.reached,
                    val.pairs, test.pairs);
        std::fflush(stdout);
    }

    const float ident_train = IdentityMse(train);
    const float power_train = NextPower(train);

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
    const PairScore first = ParallelPairMse(pool, *wm, reps, za, train);
    const double sec_eval_pre = ElapsedSec(t_eval0);

    const auto t_train = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < kEpochs; ++epoch)
    {
        wm->SetEpoch(epoch, kEpochs);
        ParallelTrainEpoch(pool, *wm, reps, za, train);
        SyncReplicas(*wm, reps);
        wm->Observe(ParallelPairMse(pool, *wm, reps, za, val).all.mse, epoch);
    }
    const double sec_train = ElapsedSec(t_train);

    const auto t_eval1 = std::chrono::steady_clock::now();
    wm->RestoreBest();
    SyncReplicas(*wm, reps);
    const PairScore train_s = ParallelPairMse(pool, *wm, reps, za, train);
    const PairScore val_s = ParallelPairMse(pool, *wm, reps, za, val);
    const PairScore test_s = ParallelPairMse(pool, *wm, reps, za, test);
    const double sec_eval = sec_eval_pre + ElapsedSec(t_eval1);

    std::printf("train mse=%.5f -> %.5f  identity=%.5f  next-power=%.5f  "
                "mse/ident=%.4g  mse/power=%.4g\n",
                static_cast<double>(first.all.mse), static_cast<double>(train_s.all.mse),
                static_cast<double>(ident_train), static_cast<double>(power_train),
                ident_train > 0.f ? static_cast<double>(train_s.all.mse / ident_train) : 0.0,
                power_train > 0.f ? static_cast<double>(train_s.all.mse / power_train) : 0.0);
    std::fflush(stdout);
    PrintScore("val", val_s);
    PrintScore("test", test_s);
    std::printf("time dataset=%.3fs  train=%.3fs  eval=%.3fs  total=%.3fs\n",
                sec_dataset, sec_train, sec_eval,
                sec_dataset + sec_train + sec_eval);
    std::printf("Look at test mse/power: you want it as low as you can get, "
                "with val mse/power about the same number. "
                "Straight vs turn is encounter vs not; keep vs change is whether "
                "a flipped (each slice uses its own next-power). Those slices do "
                "not say whether P reads a. The swap line does: the same E(x) "
                "through P with every za. If P reads a, wrong/true is well above "
                "1 and argmin==a is well above 25%%. If P ignores a, wrong/true "
                "is 1 and argmin==a is 25%%.\n");
    std::fflush(stdout);

    if (!(train_s.all.mse < first.all.mse))
        return Fail("train MSE did not fall");
    if (!(test_s.all.mse < test_s.all.power))
        return Fail("test MSE not below next-window power (zero predictor)");
    if (!(test_s.swap.mse_wrong > test_s.swap.mse_true))
        return Fail("swap: wrong heading does not score worse than the true one (P ignores a)");

    std::printf("ok\n");
    return 0;
}
