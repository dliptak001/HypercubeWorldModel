// Smoke test: Encoder, LCN + LCNTraining, Predictor train + forward,
// WorldModel encode + train, and Encode → Decode including a Decoder
// Save / Load round trip.

#include "Encoder.h"
#include "LCN.h"
#include "LCNTraining.h"
#include "Decoder.h"
#include "Predictor.h"
#include "WorldModel.h"
#include "VectorModel.h"
#include "Metrics.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int Fail(const char* what)
{
    std::printf("FAIL: %s\n", what);
    return 1;
}

int main()
{
    // --- Encoder -----------------------------------------------------------
    EncoderConfig ecfg;
    ecfg.dim = 6;
    ecfg.history_depth = 4;
    ecfg.passes = 8;
    auto enc = Encoder::Create(ecfg);
    {
        const EncoderConfig c = enc->Config();
        std::printf("[Encoder DIM=%zu M=%zu seed=%llu leak=%.3g in_scale=%.3g "
                    "out_scale=%.3g SR target=%.4f post=%.4f]\n",
                    c.dim, c.history_depth,
                    static_cast<unsigned long long>(c.seed), c.leak_rate,
                    c.input_scaling, c.output_scale, c.spectral_radius,
                    enc->RealizedSpectralRadius());
    }

    // Non-finite float knobs must be rejected even under -ffast-math.
    {
        const float nan = std::bit_cast<float>(0x7fc00000u);
        const float inf = std::bit_cast<float>(0x7f800000u);
        bool ethrew = false;
        try { EncoderConfig bad = ecfg; bad.spectral_radius = nan; Encoder::Create(bad); }
        catch (const std::invalid_argument&) { ethrew = true; }
        if (!ethrew) return Fail("Encoder NaN spectral_radius not rejected");
        ethrew = false;
        try { EncoderConfig bad = ecfg; bad.leak_rate = inf; Encoder::Create(bad); }
        catch (const std::invalid_argument&) { ethrew = true; }
        if (!ethrew) return Fail("Encoder inf leak_rate not rejected");
        ethrew = false;
        try { EncoderConfig bad = ecfg; bad.input_scaling = nan; Encoder::Create(bad); }
        catch (const std::invalid_argument&) { ethrew = true; }
        if (!ethrew) return Fail("Encoder NaN input_scaling not rejected");
        ethrew = false;
        try { EncoderConfig bad = ecfg; bad.output_scale = nan; Encoder::Create(bad); }
        catch (const std::invalid_argument&) { ethrew = true; }
        if (!ethrew) return Fail("Encoder NaN output_scale not rejected");
        ethrew = false;
        try { EncoderConfig bad = ecfg; bad.output_scale = 0.f; Encoder::Create(bad); }
        catch (const std::invalid_argument&) { ethrew = true; }
        if (!ethrew) return Fail("Encoder zero output_scale not rejected");
    }
    const size_t n = enc->Size();

    // --- LCN + LCNTraining on their own ------------------------------------
    {
        LCNConfig lcfg;
        lcfg.dim = 6;
        lcfg.z_max = 3;
        auto net = LCN::Create(lcfg);
        LCNTraining trainer(*net, LCNTrainingConfig{});
        std::vector<float> in(n, 0.5f), target(n, 1.0f);
        float first = 0.f, last = 0.f;
        for (int epoch = 0; epoch < 20; ++epoch)
        {
            trainer.SetEpoch(epoch, 20);
            trainer.ZeroGrad();
            net->Forward(in);
            last = trainer.Loss(target);
            if (epoch == 0) first = last;
            trainer.Backward();
            trainer.Adam();
        }
        std::printf("LCN N=%zu weights=%zu loss %.4f -> %.4f\n",
                    net->N(), net->Weights().size(), first, last);
        if (!(last < first)) return Fail("LCN loss did not fall");
    }

    // --- Encode → Decode ---------------------------------------------------
    // Drive the encoder with a few random fields, keep a dim-5 subcube of
    // each (compression), and train the Decoder to emit the original field.
    const size_t subcube_dim = 5;
    const size_t sub = size_t{1} << subcube_dim;
    const int samples = 8;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<std::vector<float>> fields(samples, std::vector<float>(n));
    std::vector<std::vector<float>> subcubes(samples, std::vector<float>(sub));
    std::vector<float> all_sub;
    for (int s = 0; s < samples; ++s)
    {
        for (float& x : fields[s]) x = u(rng);
        const float* out = enc->RunEpisode(fields[s]);
        for (size_t i = 0; i < sub; ++i) subcubes[s][i] = out[i];
        all_sub.insert(all_sub.end(), subcubes[s].begin(), subcubes[s].end());
    }

    // Episodes are independent: the same field gives the same output again.
    {
        const float* again = enc->RunEpisode(fields[0]);
        for (size_t i = 0; i < sub; ++i)
            if (again[i] != subcubes[0][i]) return Fail("RunEpisode not repeatable");
    }

    // output_scale is presentation: scale 2 doubles the return, not the raw cube.
    {
        EncoderConfig scfg = ecfg;
        scfg.output_scale = 2.f;
        auto enc2 = Encoder::Create(scfg);
        const float* a = enc->RunEpisode(fields[0]);
        const float* b = enc2->RunEpisode(fields[0]);
        const float* raw_a = enc->RawCube();
        const float* raw_b = enc2->RawCube();
        for (size_t i = 0; i < n; ++i)
        {
            if (raw_a[i] != raw_b[i]) return Fail("output_scale changed raw dynamics");
            if (b[i] != 2.f * a[i]) return Fail("output_scale 2 did not double the return");
        }
        const float before = enc->OutputScale();
        const float suggested = enc->SuggestOutputScale();
        if (enc->OutputScale() != before) return Fail("SuggestOutputScale mutated the Encoder");
        if (!(suggested > 0.f)) return Fail("SuggestOutputScale not positive");
    }

    {
        std::vector<float> quiet(n, 0.2f);
        if (MeanAbs(quiet) != 0.2f) return Fail("MeanAbs 0.2");
        if (Rms(quiet) != 0.2f) return Fail("Rms 0.2");
        if (MeanAbs({}) != 0.f || Rms({}) != 0.f) return Fail("MeanAbs/Rms empty");
        const float s = SuggestedOutputScale(quiet, 1.f);
        if (s != 5.f) return Fail("SuggestedOutputScale 0.2 -> 5");
        bool sthr = false;
        try { (void)SuggestedOutputScale({}, 1.f); }
        catch (const std::invalid_argument&) { sthr = true; }
        if (!sthr) return Fail("SuggestedOutputScale empty not rejected");
        sthr = false;
        try { (void)SuggestedOutputScale(std::vector<float>(n, 0.f), 1.f); }
        catch (const std::invalid_argument&) { sthr = true; }
        if (!sthr) return Fail("SuggestedOutputScale all-zero not rejected");
        enc->FitOutputScale();
        if (enc->OutputScale() != enc->SuggestOutputScale())
            return Fail("FitOutputScale did not match Suggest");
        enc->SetOutputScale(1.f);
    }

    DecoderConfig dcfg;
    dcfg.dim = 6;
    dcfg.k = subcube_dim;
    dcfg.z_max = 4;
    dcfg.gather_span = 2;
    auto dec = Decoder::Create(dcfg);
    PredictorConfig pcfg;
    pcfg.dim = subcube_dim;
    auto pred = Predictor::Create(pcfg);
    if (pred->Size() != sub)
        return Fail("Predictor Size");
    {
        float pfirst = 0.f, plast = 0.f;
        const int pepochs = 20;
        for (int epoch = 0; epoch < pepochs; ++epoch)
        {
            pred->SetEpoch(epoch, pepochs);
            pred->BeginBatch();
            float sum = 0.f;
            for (int s = 0; s < samples - 1; ++s)
                sum += pred->Accumulate(subcubes[s], subcubes[s + 1]);
            pred->EndBatch();
            if (epoch == 0) pfirst = sum;
            plast = sum;
        }
        std::printf("Predictor sub=%zu loss %.4f -> %.4f\n", sub, pfirst, plast);
        if (!(plast < pfirst)) return Fail("Predictor loss did not fall");
        const float* hat = pred->Predict(subcubes[0]);
        if (hat == nullptr) return Fail("Predictor Predict");
        bool pthrew = false;
        try { std::vector<float> bad(sub / 2, 0.f); pred->Predict(bad); }
        catch (const std::invalid_argument&) { pthrew = true; }
        if (!pthrew) return Fail("Predictor Predict short z not rejected");
        pthrew = false;
        try
        {
            std::vector<float> bad(sub / 2, 0.f);
            pred->Accumulate(bad, subcubes[0]);
        }
        catch (const std::invalid_argument&) { pthrew = true; }
        if (!pthrew) return Fail("Predictor Accumulate short z not rejected");
    }

    if (dec->CodeSize() != sub || dec->FieldSize() != n)
        return Fail("Decoder sizes");
    dec->FitInputScale(all_sub);

    float first = 0.f, last = 0.f;
    const int epochs = 30;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        dec->SetEpoch(epoch, epochs);
        dec->BeginBatch();
        float sum = 0.f;
        for (int s = 0; s < samples; ++s)
            sum += dec->Accumulate(subcubes[s], fields[s]);
        dec->EndBatch();
        if (epoch == 0) first = sum;
        last = sum;
    }
    std::printf("Decoder sub=%zu -> N=%zu scale=%.3g loss %.4f -> %.4f\n",
                sub, n, dec->InputScale(), first, last);
    if (!(last < first)) return Fail("Decoder loss did not fall");

    // --- Save / Load round trip --------------------------------------------
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "hypercube_world_model_decoder_smoke.bin";
    dec->Save(file);
    auto copy = Decoder::Load(file);
    std::filesystem::remove(file);

    if (copy->InputScale() != dec->InputScale()) return Fail("round trip input scale");
    if (copy->Config().z_max != dec->Config().z_max) return Fail("round trip z_max");
    const float* a = dec->Decode(subcubes[0]);
    const float* b = copy->Decode(subcubes[0]);
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) return Fail("round trip Decode mismatch");
    std::printf("Decoder round trip OK\n");

    // --- Validation paths --------------------------------------------------
    bool threw = false;
    try { DecoderConfig bad = dcfg; bad.k = dcfg.dim; Decoder::Create(bad); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) return Fail("k == dim not rejected");
    threw = false;
    try { DecoderConfig bad = dcfg; bad.k = dcfg.dim + 1; Decoder::Create(bad); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) return Fail("k > dim not rejected");
    threw = false;
    try { std::vector<float> zeros(sub, 0.f); dec->FitInputScale(zeros); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) return Fail("all-zero FitInputScale not rejected");

    // --- WorldModel --------------------------------------------------------
    {
        WorldModelConfig wcfg;
        wcfg.encoder = enc->Config();
        wcfg.k = subcube_dim;
        auto wm = WorldModel::Create(wcfg);
        if (wm->FieldSize() != n || wm->CodeSize() != sub)
            return Fail("WorldModel sizes");
        if (wm->K() != subcube_dim)
            return Fail("WorldModel K");
        std::vector<float> z0(sub);
        wm->Encode(fields[0], z0);
        for (size_t i = 0; i < sub; ++i)
            if (z0[i] != subcubes[0][i])
                return Fail("WorldModel Encode != k-face slice");
        std::vector<std::vector<float>> latents(samples, std::vector<float>(sub));
        for (int s = 0; s < samples; ++s)
            wm->Encode(fields[s], latents[s]);
        {
            std::vector<float> a(sub), b(sub);
            wm->Encode(fields[0], a);
            wm->Encode(fields[1], b);
            for (size_t i = 0; i < sub; ++i)
                if (a[i] != subcubes[0][i])
                    return Fail("WorldModel Encode clobbered the other buffer");
        }
        std::vector<float> a_field(n, 1.f);
        std::vector<float> za(sub);
        wm->Encode(a_field, za);
        float wfirst = 0.f, wlast = 0.f;
        const int wepochs = 20;
        for (int epoch = 0; epoch < wepochs; ++epoch)
        {
            wm->SetEpoch(epoch, wepochs);
            wm->BeginBatch();
            float sum = 0.f;
            for (int s = 0; s < samples - 1; ++s)
                sum += wm->Accumulate(latents[s], za, latents[s + 1]);
            wm->EndBatch();
            if (epoch == 0) wfirst = sum;
            wlast = sum;
        }
        std::printf("WorldModel k=%zu code=%zu loss %.4f -> %.4f\n",
                    wm->K(), wm->CodeSize(), wfirst, wlast);
        if (!(wlast < wfirst)) return Fail("WorldModel loss did not fall");

        // Save / Load with passes left at 0: each encoder resolves T from its
        // own cube (view N, action 2^k), and Load must rebuild both that way.
        {
            WorldModelConfig pcfg = wcfg;
            pcfg.encoder.passes = 0;
            auto w1 = WorldModel::Create(pcfg);
            const std::filesystem::path wfile =
                std::filesystem::temp_directory_path() / "hypercube_world_model_smoke.wm";
            w1->Save(wfile);
            auto w2 = WorldModel::Load(wfile);
            std::filesystem::remove(wfile);
            if (w2->Config().encoder.passes != w1->Config().encoder.passes)
                return Fail("WorldModel Load view passes");
            if (w2->ActionEncoderConfig().passes != w1->ActionEncoderConfig().passes)
                return Fail("WorldModel Load action passes");
            std::vector<float> picture(sub, 0.5f), c1(sub), c2(sub), v1(sub), v2(sub);
            w1->EncodeAction(picture, c1);
            w2->EncodeAction(picture, c2);
            for (size_t i = 0; i < sub; ++i)
                if (c1[i] != c2[i])
                    return Fail("WorldModel Load E(a) differs");
            w1->Encode(fields[0], v1);
            w2->Encode(fields[0], v2);
            for (size_t i = 0; i < sub; ++i)
                if (v1[i] != v2[i])
                    return Fail("WorldModel Load E(x) differs");
            const float* h1 = w1->Predict(v1, c1);
            std::vector<float> keep(h1, h1 + sub);
            const float* h2 = w2->Predict(v2, c2);
            for (size_t i = 0; i < sub; ++i)
                if (keep[i] != h2[i])
                    return Fail("WorldModel Load Predict differs");
        }
        const float* hat = wm->Predict(latents[0], za);
        if (hat == nullptr) return Fail("WorldModel Predict");
        threw = false;
        try { std::vector<float> bada(2, 0.f); wm->Predict(latents[0], bada); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("WorldModel Predict short a not rejected");
        threw = false;
        try { WorldModelConfig bad = wcfg; bad.k = wcfg.encoder.dim; WorldModel::Create(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("WorldModel k == dim not rejected");
        threw = false;
        try { WorldModelConfig bad = wcfg; bad.k = 3; WorldModel::Create(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("WorldModel k < 5 not rejected");
        threw = false;
        try
        {
            wm->SetViewOutputScale(std::bit_cast<float>(0x7fc00000u));
        }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("WorldModel NaN view output_scale not rejected");
        wm->SetViewOutputScale(2.f);
        wm->SetActionOutputScale(3.f);
        if (wm->ViewOutputScale() != 2.f || wm->ActionOutputScale() != 3.f)
            return Fail("WorldModel independent output_scales");
        {
            std::vector<float> z(sub, 1.f), a(sub, 1.f), packed(2 * sub);
            wm->Pack(z, a, packed);
            if (packed[sub] != 1.f) return Fail("WorldModel Pack is not concat");
        }
        wm->SetViewOutputScale(1.f);
        wm->SetActionOutputScale(1.f);

        // v1 files: action_scale becomes the action encoder's output_scale.
        {
            const std::filesystem::path v1file =
                std::filesystem::temp_directory_path() / "hypercube_world_model_v1.wm";
            {
                std::ofstream os(v1file, std::ios::binary);
                const char magic[4] = {'H', 'W', 'M', '1'};
                os.write(magic, 4);
                const auto wr = [&os](auto v) {
                    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
                };
                const EncoderConfig& e = wm->Config().encoder;
                wr(uint32_t{1});
                wr(static_cast<uint64_t>(e.dim));
                wr(static_cast<uint64_t>(e.seed));
                wr(e.spectral_radius);
                wr(e.leak_rate);
                wr(e.input_scaling);
                wr(static_cast<uint64_t>(e.history_depth));
                wr(static_cast<uint64_t>(wm->RequestedPasses()));
                wr(static_cast<uint64_t>(e.ic_seed));
                wr(static_cast<uint64_t>(wm->K()));
                wr(0.33f);
                wr(static_cast<uint64_t>(wm->Config().predictor.z_max));
                wr(static_cast<uint64_t>(wm->Config().predictor.gather_span));
                wr(static_cast<uint8_t>(wm->Config().predictor.tanh_last ? 1 : 0));
                wr(static_cast<uint64_t>(wm->Config().predictor.seed));
                const LCNTrainingConfig& t = wm->Config().predictor.training;
                wr(t.lr);
                wr(t.lr_min_frac);
                wr(static_cast<int32_t>(t.lr_decay_epochs));
                wr(static_cast<uint8_t>(t.restore_best ? 1 : 0));
                wr(t.beta1);
                wr(t.beta2);
                wr(t.eps);
                const auto& wts = wm->Weights();
                wr(static_cast<uint64_t>(wts.size()));
                os.write(reinterpret_cast<const char*>(wts.data()),
                         static_cast<std::streamsize>(wts.size() * sizeof(float)));
            }
            auto v1 = WorldModel::Load(v1file);
            std::filesystem::remove(v1file);
            if (v1->ViewOutputScale() != 1.f)
                return Fail("WorldModel Load v1 view output_scale");
            if (v1->ActionOutputScale() != 0.33f)
                return Fail("WorldModel Load v1 action output_scale");
        }
        threw = false;
        try { std::vector<float> bad(sub / 2); wm->Encode(fields[0], bad); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("WorldModel Encode short dst not rejected");
    }

    // --- Normaliser / Head / VectorModel -----------------------------------
    {
        std::vector<float> x = {0.f, 10.f, 2.f, 12.f, 4.f, 14.f};   // 3 x 2
        auto norm = Normaliser::Fit(x, 2, 3.f);
        if (norm.Dim() != 2) return Fail("Normaliser dim");
        std::vector<float> out(2);
        norm.Apply(std::span<const float>(x.data(), 2), out);
        if (!(out[0] >= -1.f && out[0] <= 1.f && out[1] >= -1.f && out[1] <= 1.f))
            return Fail("Normaliser Apply not in [-1,1]");
        auto again = Normaliser::FromState(norm.Mean(), norm.Std(), norm.Clip());
        std::vector<float> out2(2);
        again.Apply(std::span<const float>(x.data(), 2), out2);
        if (out[0] != out2[0] || out[1] != out2[1])
            return Fail("Normaliser FromState mismatch");
        threw = false;
        try { Normaliser::Fit(x, 2, std::bit_cast<float>(0x7fc00000u)); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("Normaliser NaN clip not rejected");
        std::printf("Normaliser dim=%zu clip=%.3g round-trip OK\n",
                    again.Dim(), again.Clip());
    }

    if (MinDim(6) != 6 || MinDim(67) != 7 || MinK(2) != 5 || MinK(38) != 6)
        return Fail("MinDim/MinK floors");
    threw = false;
    try { RequireLastDim(7, 6, "obs", "N"); }
    catch (const std::invalid_argument& e)
    {
        threw = true;
        const std::string msg = e.what();
        if (msg.find('7') == std::string::npos || msg.find('6') == std::string::npos)
            return Fail("capacity error must name both sizes");
    }
    if (!threw) return Fail("RequireLastDim 7>6 did not throw");

    {
        // Quadratic Head on a toy pairwise distance: z = [a, b], y = ||a-b||^2.
        const size_t code = 4, n = 64;
        std::vector<float> z(n * code), y(n);
        for (size_t i = 0; i < n; ++i)
        {
            const float ax = static_cast<float>(i % 8) / 7.f;
            const float ay = static_cast<float>(i / 8) / 7.f;
            const float bx = 1.f - ax, by = 0.5f;
            z[i * code + 0] = ax;
            z[i * code + 1] = ay;
            z[i * code + 2] = bx;
            z[i * code + 3] = by;
            const float dx = ax - bx, dy = ay - by;
            y[i] = dx * dx + dy * dy;
        }
        Head qh;
        qh.Fit(z, code, y, n);
        const Head::Score qs = qh.ScoreOn(z, y);
        std::printf("Head quadratic toy-distance R2 %.4f\n", qs.r2);
        if (!(qs.r2 > 0.99f)) return Fail("Head quadratic R2 not near 1");
        Head lh(Head::Kind::Linear);
        lh.Fit(z, code, y, n);
        if (!lh.Fitted()) return Fail("Head linear did not fit");
        threw = false;
        try { std::vector<float> dummy(n); qh.Apply(z, dummy, z); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("Head za omitted path accepted za");
    }

    {
        WorldModelConfig wcfg;
        wcfg.encoder.dim = 6;
        wcfg.encoder.history_depth = 4;
        wcfg.encoder.passes = 8;
        wcfg.k = 5;
        auto vm = VectorModel::Create(WorldModel::Create(wcfg));
        const float lo[2] = {-1.f, -1.f}, hi[2] = {1.f, 1.f};
        vm->SetActionBounds(lo, hi);
        const size_t c = vm->CodeSize();
        std::vector<float> obs{0.2f, -0.3f}, act{0.1f, -0.2f};
        std::vector<float> z(c), za(c);
        vm->Encode(obs, z);
        vm->EncodeAction(act, za);
        const float* hat = vm->Predict(z, za);
        if (hat == nullptr) return Fail("VectorModel Predict");
        std::vector<float> path(2 * c);
        vm->Rollout(z, act, path);   // H=1
        if (path[0] != z[0]) return Fail("VectorModel Rollout z0");

        Head dist(Head::Kind::Linear);
        std::vector<float> y{0.1f};
        dist.Fit(z, c, y, 1);
        threw = false;
        try { vm->SetHead("empty", Head()); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) return Fail("VectorModel SetHead unfitted not rejected");
        vm->SetHead("dist2", dist);
        std::vector<float> cost(1);
        vm->Cost(path, 1, 2, cost);

        const std::filesystem::path vfile =
            std::filesystem::temp_directory_path() / "hypercube_vector_model_smoke.hvm";
        vm->Save(vfile);
        auto vm2 = VectorModel::Load(vfile);
        std::vector<float> z2(c);
        vm2->Encode(obs, z2);
        for (size_t i = 0; i < c; ++i)
            if (z[i] != z2[i])
                return Fail("VectorModel Load Encode differs");
        if (!vm2->GetHead("dist2") || vm2->GetHead("dist2")->Bias() != dist.Bias())
            return Fail("VectorModel Load Head");
        if (!vm2->ObsNormaliser() && vm->ObsNormaliser())
            return Fail("VectorModel Load norm");
        auto nfit = Normaliser::Fit(obs, 2);
        vm->SetObsNormaliser(nfit);
        vm->Save(vfile);
        vm2 = VectorModel::Load(vfile);
        if (!vm2->ObsNormaliser() || vm2->ObsNormaliser()->Mean()[0] != nfit.Mean()[0])
            return Fail("VectorModel Load Normaliser mean");

        threw = false;
        try { WorldModel::Load(vfile); }
        catch (const std::runtime_error& e)
        {
            threw = true;
            const std::string msg = e.what();
            if (msg.find("magic") == std::string::npos)
                return Fail("WorldModel::Load of VectorModel file should be bad magic");
        }
        if (!threw) return Fail("WorldModel::Load of HVM1 did not throw");
        std::filesystem::remove(vfile);

        const std::filesystem::path wfile =
            std::filesystem::temp_directory_path() / "hypercube_vector_model_bare.wm";
        vm->World().Save(wfile);
        auto wm_only = VectorModel::Load(wfile);   // HWM1 as wm-only
        if (wm_only->GetHead("dist2") != nullptr)
            return Fail("VectorModel Load HWM1 should have no heads");
        WorldModel::Load(wfile);   // still a WorldModel file
        std::filesystem::remove(wfile);

        const float mse = NoChangeMse(z, za);
        const float ratio = OneStepRatio(mse, z, za);
        std::printf("metrics no-change %.5f one-step-ratio %.3f\n", mse, ratio);
        auto lin = LinearR2On(z, obs, z, obs, 1, 1, c, 2);
        std::printf("metrics linR2 min %.3f mean %.3f\n", lin.min, lin.mean);
        auto asens = MeasureActionSensitivity(*vm, z, 1, mse > 0.f ? mse : 1.f, 1, 1);
        std::printf("metrics act %.3f\n", asens.act);
        std::vector<float> z_true(path);
        std::fill(z_true.begin() + static_cast<std::ptrdiff_t>(c), z_true.end(), 0.f);
        auto re = RolloutErrorFromCodes(path, z_true, 1, 2, c);
        std::printf("metrics rollout-error h1 %.5f ratio %.3f\n", re.error[0], re.ratio[0]);
        std::printf("VectorModel N=%zu code=%zu round-trip OK\n",
                    vm->FieldSize(), vm->CodeSize());
    }

    std::printf("all OK\n");
    return 0;
}
