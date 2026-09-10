# World model

**Status: implemented in WorldModel/. Pack is E(x) cat E(a). E(a) comes from a second Encoder on a k-cube. Terrain walker is tests/TerrainWalker/. The SDK guide is [CPP_SDK.md](CPP_SDK.md).**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| WorldModel | This component: two frozen Encoders plus a trained Predictor. The public product face. |
| Encoder | Frozen hypercube reservoir in Encoder/, specified in [encoder.md](encoder.md). Perception. Weights never move. The view encoder sits on a dim-cube. |
| action encoder | A second Encoder with the same knobs and seeds but a k-cube. Its whole output is E(a). No cut on this side. |
| Predictor | LCN in Predictor/, specified in [predictor.md](predictor.md). Dynamics. Here its cube is k+1, not k. |
| Decoder | Reconstruction meter in Decoder/. Not part of this object. |
| dim | Dimension of the encoder cube. |
| N | Encoder vertices and window length, N = 2ᵈⁱᵐ. |
| k | Dimension of the kept face, and of the action cube. Strictly less than dim, and at least 5. |
| k-face | The low 2ᵏ vertices of an encoder episode. The latent. |
| field, window | One observation: N values, vertex i holding sample i. What Encode reads. |
| channel | One code of length 2ᵏ on the Predictor cube. Channel 0 is E(x). Channel 1 is E(a). Pack concatenates. More channels later mean k+c slots. |
| E(x) | Encode(x, z): run an episode, write the k-face into z. |
| a | The action, painted as a picture of 2ᵏ cells, one per vertex of the action cube. What the picture looks like is the caller's job. |
| E(a) | EncodeAction(a, za): run an action episode on the k-cube, write its whole output. Length 2ᵏ. |
| za | The buffer holding E(a). Length 2ᵏ. Second argument of Predict and Accumulate. |
| action_scale | Multiplier on E(a) as Pack lays it on the extra bit-face. za itself is untouched. Brings E(a) to the level of E(x). |
| first subcube | Predictor vertices whose extra address bit is 0. Length 2ᵏ. E(x) in, predicted next E(x) out. |
| extra bit-face | Predictor vertices whose extra address bit is 1. Length 2ᵏ. Holds E(a). |
| ŝ | Predict(z, za): a predicted next k-face. The first subcube of the Predictor output. |
| identity | Using zₜ itself as the guess for zₜ₊₁. See [jepa_predictor_test.md](jepa_predictor_test.md). |
| z_max | Predictor LCN depth. 0 means k+1. |
| gather_span | Predictor LCN lookback window width in fields. |
| seed | Predictor LCN weight seed. The Encoder has its own seed and ic_seed. |
| batch | One BeginBatch, some Accumulate calls, one EndBatch. |
| Pack | Concatenate two k-faces: E(x) on the first subcube, action_scale × E(a) on the extra bit-face. |
| LastCube | Full encoder episode from the most recent Encode. Length N. |
| Rollout | Predict chained over H action codes: Rollout(z0, actions, out) writes H + 1 view codes, the first being z0. |
| H | Number of action codes in a rollout. |
| PaintStripes | Free function. Lays a short vector onto a field as contiguous stripes, so a thin state or action vector becomes a fat picture. |
| Save, Load | Config and Predictor weights to and from a file. The encoders are rebuilt from their seeds; passes is written as given, so 0 stays 0. Adam state is not saved. |

This is a world model with frozen perception. The view Encoder turns
a field of N values into a k-face and does not learn. The action
Encoder is the same kind of reservoir on a smaller cube, one of
dimension k, and does not learn either. The Predictor learns how the
view code moves given the action code. The JEPA picture is the same
diagram, guess the next code rather than the next samples, but the
encoders are not trained by that loss, so the class is not named JEPA.

The Decoder is a separate meter: it asks whether the k-face still
carries the field. It does not sit in this loop.

## Two steps

Perception and dynamics are separate so a stream can be encoded once
and trained many times.

```
wm.Encode(x, z);                     // x: N floats. z: 2ᵏ floats, caller-owned.
wm.EncodeAction(a_field, za);        // a_field and za: 2ᵏ floats.
const float* hat = wm.Predict(z, za); // hat: 2ᵏ floats.
```

Encode always runs a full view episode (N values) and writes the
low k-face into a buffer the caller owns. That cut is the compression.
EncodeAction runs a full action episode (2ᵏ values) and writes all of
it: the action cube is already the k-face size, so there is nothing
to cut. WorldModel does not keep a code: two windows need two buffers.
The action field is another encoder episode, not a side door.

The Predictor never sees dim. Its own cube is k+1, twice the k-face.
Pack lays E(x) on the first subcube and action_scale × E(a) on the
extra bit-face. Vertex i of the view sits one hop from vertex i of
the action channel. Nothing is added into E(x). Predict returns the
first subcube of that cube. If P ignores the second channel, two
different actions from the same view produce the same guess. They
must not.

## The action path

The action goes through the same kind of machine as the view, on a
cube sized to fit the code it has to produce.

1. **The caller paints the action as a picture.** The picture has 2ᵏ
   cells, one per vertex of the action cube. What it shows is the
   task's business: a constant fill when there is no motor, a
   half-plane toward the heading in the terrain walker. It should be
   fat, meaning many cells carry signal, so the reservoir has
   something to work with.

2. **The action encoder runs one episode on it.** This encoder is
   built from the view encoder's EncoderConfig with dim replaced by k:
   same leak, same input scaling, same spectral radius target, same
   passes, same seeds. Only the cube is smaller, so the weight draw is
   a different draw of the same kind. It is frozen like the view
   encoder.

3. **Its whole output is E(a).** The action cube has exactly 2ᵏ
   vertices, so its output is already the length of the view's
   k-face. Nothing is cut and nothing is padded. That is why the
   action cube is k and not dim: the code has to sit next to E(x) on
   the Predictor, and 2ᵏ is the size that fits.

4. **Encode once per distinct action.** An action code depends only
   on the picture, so a task with a handful of actions encodes each
   picture once and reuses the code for every pair.

5. **Pack scales it on the way in.** The whole output of a small cube
   runs hotter than a face cut from a big one. action_scale is one
   multiplier applied as Pack lays E(a) on the extra bit-face, so P
   sees the two channels at a chosen ratio. The stored code za is
   never changed; only what P sees is scaled.

6. **P reads it one hop away.** On the (k+1)-cube, vertex i of E(a)
   is the neighbour of vertex i of E(x) along the extra axis. The
   LCN's first depth already sees both. The loss never mentions the
   action half; P is free to use it or not, and the swap check in
   [terrain_walker.md](terrain_walker.md) is how you find out which.

Train on pairs of view codes and the action code that took you from
one to the next, not on fields:

```
wm.Encode(x_t, z_t);
wm.Encode(x_next, z_next);
wm.EncodeAction(a_field_t, za_t);
wm.BeginBatch();
for each pair: wm.Accumulate(z_t, za_t, z_next);
wm.EndBatch();
```

Accumulate packs E(x) cat E(a), forwards, and takes loss on the first
subcube only. The extra bit-face is unconstrained. SetEpoch, Observe,
and RestoreBest are the Predictor's schedule and best-weight
snapshot. ResetTraining forgets Adam; it does not touch the Encoder.

A task with no motor still runs some action picture through the
action encoder. The two-sine mix in
[world_model_test.md](world_model_test.md) uses a constant fill. The
Terrain walker in [terrain_walker.md](terrain_walker.md) paints a
half-plane toward the executed cardinal on a 2ᵏ strip.

## Files

```
WorldModel/
    WorldModel.h        WorldModelConfig, class WorldModel
    WorldModel.cpp
```

CMake target WorldModel, a static library linking Encoder and
Predictor. Nothing in WorldModel includes Decoder.

## Interface

```cpp
struct WorldModelPredictorConfig
{
    size_t   z_max;        // 0 → k+1
    size_t   gather_span;
    bool     tanh_last;
    uint64_t seed;
    LCNTrainingConfig training;
};

struct WorldModelConfig
{
    EncoderConfig encoder;
    size_t   k;            // code face and action cube; in [5, encoder.dim)
    float    action_scale; // multiplier on E(a) at Pack; > 0
    WorldModelPredictorConfig predictor;
};

void PaintStripes(std::span<const float> src, std::span<float> dst);

class WorldModel
{
public:
    static std::unique_ptr<WorldModel> Create(const WorldModelConfig& cfg);
    static std::unique_ptr<WorldModel> Load(const std::filesystem::path& file);
    void Save(const std::filesystem::path& file) const;

    const float* Encode(std::span<const float> field, std::span<float> dst);
    const float* LastCube() const;       // full view episode; until next Encode
    const float* EncodeAction(std::span<const float> field, std::span<float> dst);
    const float* Predict(std::span<const float> z, std::span<const float> a);
    void Rollout(std::span<const float> z0, std::span<const float> actions,
                 std::span<float> out);  // H codes in, H + 1 codes out
    size_t RequestedPasses() const;      // encoder.passes as given; what Save writes
    void Pack(std::span<const float> z, std::span<const float> a,
              std::span<float> dst) const;

    void  BeginBatch();
    float Accumulate(std::span<const float> z, std::span<const float> a,
                     std::span<const float> next);
    void  EndBatch();

    void  SetEpoch(int epoch, int num_epochs = 0);
    void  Observe(float metric, int epoch);
    void  RestoreBest();
    void  ResetTraining();

    const std::vector<float>& Weights() const;
    void  LoadWeights(std::span<const float> w);
    const std::vector<float>& Grad() const;
    void  AddGrad(std::span<const float> g);

    static constexpr const char kVersion[];
    size_t FieldSize() const;            // N
    size_t CodeSize() const;             // 2ᵏ; also the action field length
    size_t K() const;
    const WorldModelConfig& Config() const;
    EncoderConfig ActionEncoderConfig() const;   // encoder with dim = k
    float RealizedSpectralRadius() const;
    float ActionRealizedSpectralRadius() const;
};
```

Validation at Create: k must be in [5, encoder.dim), plus everything
Encoder and Predictor already check. The action encoder is the view
EncoderConfig with dim replaced by k, not a knob; 5 is the smallest
cube Encoder accepts. action_scale must be > 0. Predictor dim is
k+1, not a knob. Encode rejects a dst that is not CodeSize() long.
EncodeAction rejects a field or dst that is not CodeSize() long.
Pack rejects z or a that is not CodeSize() long, and a dst that is
not 2 × CodeSize() long. Rollout rejects an actions span that is not
a whole number of codes and an out span that is not one code longer.
PaintStripes rejects an empty source or one longer than its
destination.

Save writes the config and the Predictor weights; Load reads them,
rebuilds both encoders from the seeds in the config, and checks the
weight count. Adam state is not saved. Encode, EncodeAction, and
Predict on the reloaded instance reproduce the original exactly. The
main smoke test checks all three across a Save and Load with passes
left at 0.

Passes needs care because 0 means a full tour of whichever cube the
encoder sits on: the view encoder resolves it to N and the action
encoder to 2ᵏ. Save writes passes as it was given, 0 included, so
Load resolves it per cube the way Create did. Config() is the
resolved snapshot and reports the view encoder's T, so a config
rebuilt from Config() would give the action encoder N passes and a
different E(a). Keep the config you built, or use Save; do not
rebuild from Config(). RequestedPasses() returns the value as given.

Same-song scoring without a motor is [jepa_predictor_test.md](jepa_predictor_test.md)
(Predictor on the k-face directly). Many two-sine draws (train a mix,
score held, constant dummy E(a)) is [world_model_test.md](world_model_test.md).
The number is test mse/power, not the word ok.
