# HypercubeWorldModel C++ SDK

HypercubeWorldModel is a **world model with frozen encoders**. Its
neurons sit on the vertices of a Boolean hypercube: a cube of
dimension dim has N = 2ᵈⁱᵐ vertices and one neuron on each. A view,
meaning one observation, comes in as a field of N values, one per
vertex. A frozen hypercube reservoir, the Encoder, runs an episode on
that field and returns a cube of N values, and the first 2ᵏ of them
are kept, for some k smaller than dim. Those 2ᵏ values are the view's
code, written E(x). It is what other systems call the latent. An
action, meaning what the agent did, comes in as a picture of 2ᵏ
values and goes through a second frozen Encoder on a cube of
dimension k; its whole output is the action code, E(a). A locally
connected net, the Predictor, learns to map E(x) and E(a) to the next
E(x). It is the only part of the world model that trains.

Three classes own the product. WorldModel is the world model: the two
encoders and the Predictor. Decoder is reconstruction: a second
locally connected net that takes a code of 2ᵏ values and gives back a
field of N, trained on its own. VectorModel is the planner surface for
vector hosts: optional Normalisers, PaintStripes, encode from short
vectors, raw-action rollout, named Heads, and capacity checks. A host
that already has a field uses WorldModel. A task that has to see what
a code stands for needs the Decoder. Headers WorldModel.h, Decoder.h,
and VectorModel.h; plain C++23, no dependencies beyond the standard
library.

This is a **map API**, not a stream API: one field in, one episode, one
code out. The model does not remember the last view. State that has to
persist across steps lives with the caller.

The same product from Python: **[Python_SDK.md](Python_SDK.md)**.  
The components, one by one: [encoder.md](encoder.md) / [lcn.md](lcn.md) / [predictor.md](predictor.md) / [world_model.md](world_model.md) / [decoder.md](decoder.md).  
Worked programs: [examples/quick_start.cpp](../examples/quick_start.cpp)
(WorldModel) and [examples/vector_start.cpp](../examples/vector_start.cpp)
(VectorModel).

## Contents

- [Build and link](#build-and-link)
- [Quick start](#quick-start)
- [What a step is](#what-a-step-is)
- [API reference](#api-reference)
- [The Decoder](#the-decoder)
- [VectorModel](#vectormodel)
- [Input data layout](#input-data-layout)
- [Driving the model from a planner](#driving-the-model-from-a-planner)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Build and link

Requirements: a **C++23** compiler (GCC 13+, Clang 17+, MSVC 2022+) and
**CMake 3.21 or later**. The library targets are **WorldModel**,
**Decoder**, and **VectorModel**. WorldModel carries Encoder, LCN, and
Predictor with it; Decoder carries LCN; VectorModel carries WorldModel,
Normaliser, and Head. Each exports its own directory as a public
include directory.

From your own project:

```cmake
add_subdirectory(path/to/HypercubeWorldModel)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE WorldModel Decoder VectorModel)
```

```cpp
#include "WorldModel.h"
#include "Decoder.h"
#include "VectorModel.h"
```

Leave Decoder out if you do not reconstruct. Leave VectorModel out if
you paint your own fields. The smoke test and the two examples build
alongside.

Building this repo directly (CLion: open, reload CMake, build, or any
shell with the toolchain available):

```bash
cmake --build cmake-build-release
```

Use **Release** for real runs. Release builds compile with fast-math and
the optimisation flags shared across the family, and the Python wheel
will compile the same core the same way.

### Executables

| Binary | Role |
|--------|------|
| quick_start | The WorldModel program below, compiled so this page stays true |
| vector_start | VectorModel sibling: short obs, raw-action rollout, Head |
| HypercubeWorldModel | Smoke test of every component |

Every knob in these programs is a constant at the top of a source file;
there are no command-line arguments.

## Quick start

A complete toy task, verified against the library:
[examples/quick_start.cpp](../examples/quick_start.cpp), built by the
quick_start target. A point on a plane moves by a bounded velocity. The
view is the position, the action is the velocity. Its printed test
error is about one percent of the identity guess, and it checks that a
saved and reloaded model encodes and predicts the same. The main
smoke test, main.cpp, is the worked Decoder program: it encodes,
cuts, trains a Decoder to give the fields back, and round-trips it
through Save and Load.

### The cycle

Every real run is the loop in that program, with restore-best around it:

```text
fill WorldModelConfig
WorldModel::Create once
Encode every view and EncodeAction every action once, into your storage
for each epoch:
    SetEpoch                     (cosine learning rate)
    for each batch:
        BeginBatch
        per pair: Accumulate(z, za, next)
        EndBatch
    score the validation pairs with Predict
    Observe(validation metric, epoch)
RestoreBest
Predict / Rollout / Save
```

Encoding is outside the epoch loop on purpose. The encoders are frozen,
so a code never changes; a stream is encoded once and trained on many
times.

## What a step is

```text
view field (N values)                 action picture (2ᵏ values)
    |                                     |
    v  view encoder, one episode          v  action encoder, one episode
cube of N values                      cube of 2ᵏ values
    |  keep the first 2ᵏ                  |  keep all of it
    v                                     v
E(x), the view code ------+   +------ E(a), the action code
                          v   v
        (k+1)-cube: E(x) on one half, E(a) on the other
                          |
                          v  Predictor, one forward pass
        2ᵏ⁺¹ outputs; the first 2ᵏ are the predicted next E(x)
```

- The view encoder runs on the dim-cube; the action encoder is the same
  kind of machine on a k-cube, built from the same knobs with dim
  replaced by k.
- Both encoders are frozen. Nothing in them is ever trained, and a
  code depends only on its field.
- The Predictor's cube is k+1, twice the code. Vertex i of E(a) sits one
  hop from vertex i of E(x), so the first depth of the net already sees
  both. Only the first 2ᵏ outputs are read; the loss is taken there
  and nowhere else.
- Nothing in the loss mentions the action. Whether the net reads it is
  a question for a test: feed one view code with two different action
  codes and see whether the prediction changes.
- The Decoder stands outside this diagram. It takes any view code, the
  encoder's or the Predictor's, and gives back a field of N values.

## API reference

Authoritative signatures and contracts live in **WorldModel.h**, which
is fully doc-commented. This section is the host-oriented map.

### Configuration

Everything is fixed at Create, which validates and throws
std::invalid_argument on violations. WorldModelConfig carries an
EncoderConfig for the view encoder and a WorldModelPredictorConfig
for the Predictor.

```cpp
struct EncoderConfig {
    size_t   dim;               // cube dimension; N = 2^dim; in [5, 24]; a WorldModel needs at least 6
    uint64_t seed;              // weight draw
    float    spectral_radius;   // target for the recurrent block; finite, > 0
    float    leak_rate;         // leaky integrator mix; finite, in (0, 1], 1 = full replacement
    float    input_scaling;     // input drive strength; finite
    size_t   history_depth;     // delay line length M; in [1, 64]
    size_t   passes;            // passes per episode T; 0 = a full tour of the cube
    uint64_t ic_seed;           // episode start state s0; separate from seed
    float    output_scale;      // presentation gain on the returned cube; finite, > 0; default 1
};

struct LCNTrainingConfig {
    float lr;                   // Adam step size; cosine peak; finite, > 0
    float lr_min_frac;          // floor = lr * lr_min_frac; in [0, 1]; 1 = constant
    int   lr_decay_epochs;      // cosine horizon; 0 = the num_epochs given to SetEpoch
    bool  restore_best;         // snapshot weights on a new low metric
    float beta1;                // Adam first-moment decay; in [0, 1)
    float beta2;                // Adam second-moment decay; in [0, 1)
    float eps;                  // Adam denominator floor; finite, > 0
};

struct WorldModelPredictorConfig {
    size_t   z_max;             // depth; 0 = use k+1, else >= 2
    size_t   gather_span;       // lookback window width; in [2, 6]
    bool     tanh_last;         // true: tanh on the last depth too
    uint64_t seed;              // weight draw; the encoders have their own
    LCNTrainingConfig training; // Adam and cosine schedule
};

struct WorldModelConfig {
    EncoderConfig encoder;                 // the view encoder; the action encoder is the same with dim = k
    size_t   k;                            // kept face and action cube; in [5, encoder.dim)
    WorldModelPredictorConfig predictor;   // dim is always k+1, not a knob
};
```

predictor.z_max, gather_span, tanh_last, seed, and training are the
Predictor's LCN and trainer; its dim is not a knob because it is
always k+1.

What the encoder knobs do to an episode is in [encoder.md](encoder.md).
What the training knobs do to a run is in [predictor.md](predictor.md).

| Knob | Guidance |
|------|----------|
| encoder.dim | Sized to the view. A view shorter than N goes through PaintStripes |
| k | Chooses the code length, 2ᵏ. Smaller is more compression |
| encoder.output_scale | Presentation gain on the returned cube. Default 1. View and action encoders can be set independently after Create. |
| predictor.z_max | Depth buys reach on the Predictor cube. 0 means k+1, antipodal reach on that cube |
| predictor.training.lr, batch size | The gradient is a batch **sum**; scale them together |
| predictor.training.restore_best | Default on. Feed Observe a validation metric |

### Sizes

```cpp
wm->FieldSize();    // N = 2^dim, length of a view field
wm->CodeSize();     // 2^k, length of a code, of an action picture, and of E(a)
wm->K();
WorldModel::kVersion; // library version string, "1.1.0"
wm->Config();       // as given (passes of 0 stays 0); predictor.z_max is resolved
```

### Encode

Codes live in buffers you own. By convention z holds E(x) and za
holds E(a); both are CodeSize() long.

```cpp
std::vector<float> z(wm->CodeSize()), za(wm->CodeSize());
wm->Encode(field, z);          // field: N floats. Writes E(x) into z.
wm->LastCube();                // scaled full N-float episode behind the last Encode
wm->LastRawCube();             // unscaled delay-line cube; a span of this may be passed to Encode
wm->EncodeAction(picture, za); // picture: 2^k floats. Writes E(a) into za.

PaintStripes(short_vector, field);    // short vector in, field out
PaintStripes(short_vector, picture);  // works for the action picture too
```

Encode and EncodeAction write into buffers you own and return that
buffer's data pointer. The model keeps no code.

### Predict and Rollout

Predict takes one pair and writes the predicted next view code into
a buffer you own, same as Encode. Rollout chains Predict over H
action codes, each already through EncodeAction, and writes H + 1
view codes, the first being the start.

```cpp
wm->Predict(z, za, hat);                 // hat: 2^k floats you own
wm->Rollout(z0, actions, out);           // actions: H codes end to end. out: H + 1 codes, out[0] = z0
wm->Pack(z, za, packed);                 // what the Predictor sees, 2^(k+1) floats; rarely needed
```

### Train

A batch is one BeginBatch, some Accumulate calls, and one EndBatch.
An epoch is one pass over the training pairs; SetEpoch sets the
cosine learning rate for it. With restore_best on, Observe snapshots
the Predictor weights on a new low metric and RestoreBest writes them
back.

```cpp
wm->BeginBatch();                         // start of each batch
float e = wm->Accumulate(z, za, next);    // one pair; returns 0.5 * SSE over the code
wm->EndBatch();                           // one Adam step per batch

wm->SetEpoch(epoch, num_epochs);          // cosine lr; once per epoch
wm->Observe(metric, epoch);               // restore_best: lower wins, strict
wm->RestoreBest();                        // write the snapshot back
wm->ResetTraining();                      // forget Adam, lr, best; keep weights

const std::vector<float>& w = wm->Weights();   // Predictor weights
wm->LoadWeights(w2);                           // exact-length replacement
wm->Grad();  wm->AddGrad(g);                   // sum a replica gradient onto the master; see Limitations
```

### Persist

```cpp
wm->Save("model.wm");
auto again = WorldModel::Load("model.wm");
wm->RequestedPasses();          // encoder.passes as given to Create; what Save writes
```

## The Decoder

The encoder keeps 2ᵏ of its N values. The Decoder is a locally
connected net on the dim-cube that takes those 2ᵏ values, places them
on its first 2ᵏ vertices with every other vertex fed zero, and writes
a field of N values: the reconstruction. Training pairs are a code
and the field it came from. Once trained, it decodes any code of 2ᵏ
values, whether Encode produced it or Predict did. It is a separate
class with its own weights, its own training cycle, and its own file,
and it never touches the WorldModel.

Pairing it with a WorldModel is two numbers: its dim is the
encoder's dim and its k is the WorldModel's k. Everything else is its own.

```cpp
struct DecoderConfig {
    size_t   dim;               // output cube; the encoder's dim
    size_t   k;                 // input face; the WorldModel's k; strictly less than dim
    size_t   z_max;             // depth; 0 = use dim, else >= 2
    size_t   gather_span;       // lookback window width; in [2, 6]
    bool     tanh_last;         // true: tanh on the last depth too
    uint64_t seed;              // weight draw
    LCNTrainingConfig training; // Adam and cosine schedule
};
```

```cpp
DecoderConfig dcfg;
dcfg.dim = wm->Config().encoder.dim;
dcfg.k = wm->K();
auto dec = Decoder::Create(dcfg);

dec->CodeSize();               // 2^k, same as wm->CodeSize()
dec->FieldSize();              // N, same as wm->FieldSize()

dec->FitInputScale(all_codes); // once, over the training codes laid end to end
dec->Decode(z, field);             // field: N floats you own

dec->BeginBatch();
float e = dec->Accumulate(z, field); // one pair: code in, the field it came from as target
dec->EndBatch();
dec->SetEpoch(epoch, num_epochs);  dec->Observe(metric, epoch);  dec->RestoreBest();
dec->ResetTraining();

dec->Save("model.dec");
auto again = Decoder::Load("model.dec");   // config, input scale, and weights come back

const std::vector<float>& w = dec->Weights();   // depth, axis, tap, vertex
dec->LoadWeights(w2);  dec->Grad();  dec->AddGrad(g);   // as on WorldModel
```

The cycle is the WorldModel's cycle with FitInputScale before the
first epoch. FitInputScale sets one constant so the scaled codes lie
in [−1, 1]; it is frozen after that and saved with the weights. The
loss covers the whole field, not a prefix. What the knobs do, and how
the reconstruction error behaves as k moves, is in
[decoder.md](decoder.md).

## VectorModel

Authoritative signatures live in **VectorModel.h**, **Normaliser.h**,
**Head.h**, and **Metrics.h**. This is the host-oriented map. A host
that already has a field uses WorldModel and skips this class.

Create(cfg) builds and owns a WorldModel. Create(unique_ptr) takes
ownership of one you already have. Attach is a non-owning view
(the WorldModel must outlive the VectorModel). Load reads an `HVM1`
file, or a bare `HWM1` as wm-only.

```cpp
auto vm = VectorModel::Create(cfg);                       // builds and owns the WorldModel
auto vm2 = VectorModel::Create(WorldModel::Create(cfg));  // same, if you already have a WorldModel
auto view = VectorModel::Attach(*wm);                     // wm must outlive view
auto again = VectorModel::Load("model.hvm");
vm->Save("model.hvm");

size_t MinDim(obs_dim);   // max(6, ceil(log2(obs_dim))); floor, not a config
size_t MinK(act_dim);     // max(5, ceil(log2(act_dim)))
```

Optional attachments: obs/act Normaliser, action bounds the planner
clips against, named fitted Heads. SetHead rejects an unfitted Head.
obs_dim / act_dim are noted on the first encode, or set explicitly.

```cpp
Normaliser n = Normaliser::Fit(x, dim, /*clip=*/3.f);   // (count × dim) row-major
n.Apply(x, dst);                                        // into [-1, 1]
vm->SetObsNormaliser(n);
vm->SetActionBounds(low, high);
vm->SetObsDim(obs_dim);
vm->SetActDim(act_dim);

Head h(Head::Sign::Cost);                                // LCN on the code cube
h.Fit(z, code_size, y);                                 // count is y.size(); za omitted is the default
h.Predict(z, dst);                                      // one scalar per code
Head::Score s = h.ScoreOn(z, y);                        // R²; AUC only if y is strictly 0/1 both classes
h.PlanCost(zs, out);                                    // in-place over zs; extra RAM is one packed cube
vm->SetHead("dist2", h);
```

Encode / EncodeAction / Rollout take **raw** short vectors, not fields
or codes. Rollout is one plan (H × act_dim); last-dim must match
act_dim once that is set. Cost uses the single attached Head's
PlanCost, or L2 of the last code to a goal.

```cpp
vm->Encode(obs, z);                 // optional norm, PaintStripes, WorldModel::Encode
vm->EncodeAction(a, za);
vm->Predict(z, za, hat);
vm->BeginBatch(); vm->Accumulate(z, za, next); vm->EndBatch();  // same cycle as WorldModel
vm->Rollout(z0, actions, out);      // actions: H × act_dim; out: (H+1) × code
vm->Cost(zs, out);                  // one Head, or pass goal_z of CodeSize()
```

Health metrics are free functions, not methods. RankMe (SVD) is not
here.

```cpp
NoChangeMse(z, zn);
OneStepRatio(mse, z, zn);
MeasureActionSensitivity(*vm, z, count, mse, seed);   // samples in action bounds if set
LinearR2On(z, y, z_val, y_val, train_n, val_n, z_dim, y_dim);
RolloutErrorFromCodes(z_pred, z_true, windows, path_len, code_size);  // path_len is H+1
```

Worked program: [examples/vector_start.cpp](../examples/vector_start.cpp).

## Input data layout

- **A view field** is N floats, one per vertex of the view cube.
- **An action picture** is 2ᵏ floats, one per vertex of the action cube.
- **Short vectors.** A state vector or an action vector may have only
  a handful of values, while the cube has N cells or 2ᵏ cells. Writing
  those few values into the first few cells and leaving the rest zero
  would give the encoder almost nothing to respond to. PaintStripes
  instead divides the cube into as many equal blocks as there are
  values and fills each block with one value, so the whole cube
  carries the vector.
- **Your own picture.** Anything that writes N floats, or 2ᵏ floats,
  is a valid painter.
- **Codes** are 2ᵏ floats and live in caller storage. The same buffer
  feeds Predict and Decode.
- Every value in a field, a picture, or a code is a float. Lengths and
  indices are size_t.

## Driving the model from a planner

**VectorModel** is the planner surface for short-vector hosts.
WorldModel.Rollout takes action **codes**; VectorModel.Rollout takes
**raw** actions. Scoring is a named Head, or L2 to a goal code.
WorldModel never sees action bounds.

A sampling planner maps onto the SDK like this:

| Planner needs | SDK | Note |
|---------------|-----|------|
| latent_dim | CodeSize() | 2ᵏ |
| encode(obs) | VectorModel::Encode | optional Normaliser, PaintStripes, then WorldModel::Encode. obs last-dim vs N is a hard check |
| encode_action(a) | VectorModel::EncodeAction | optional act Normaliser, PaintStripes onto code_size |
| predict(z, za) | Predict | one step, one pair |
| rollout(z0, actions) | VectorModel::Rollout | raw (H × act_dim); H + 1 codes out, the first is z0 |
| action bounds | VectorModel action low/high | the planner clips here; the WorldModel never sees bounds |
| cost(zs, goal) | VectorModel::Cost / Head::PlanCost | a fitted Head, or L2 of the last code to a goal code |
| cube floor | MinDim(obs_dim), MinK(act_dim) | capacity floor, not a config |

A host that already has a field uses WorldModel and PaintStripes
itself. VectorModel always paints stripes; there is no painter hook.

A goal is a view like any other: encode it and compare codes, or fit a
Head on a host y. Because the model is a map with no memory, the same
instance serves every planner in turn. A planner that wants predictions
on many threads uses the replica recipe under Limitations.

Worked program: [examples/vector_start.cpp](../examples/vector_start.cpp).
Do not name this surface after a host suite.

Other surfaces fit the same way. A gym-style environment supplies obs
and action vectors; a stream supplies windows; an image supplies
pixels. None of them are known to the SDK, and none need to be.

## Error handling

Create validates every config range and throws std::invalid_argument;
so do Encode, EncodeAction, Predict, Rollout, Accumulate, and
LoadWeights on bad lengths, and PaintStripes on an empty or oversized
source. Decoder::Create, Decode, Accumulate, and FitInputScale do the
same. Save and Load, on both classes, throw std::runtime_error on
file trouble.

Typical mistakes:

| Symptom / assumption | Fix |
|----------------------|-----|
| Create throws on k | k must be at least 5 and strictly less than encoder.dim. So encoder.dim must be at least 6: an Encoder accepts dim 5, but a WorldModel on it has no room for a k |
| SetViewOutputScale / SetActionOutputScale throws | The scale must be finite and greater than zero |
| Create throws on an encoder float knob | spectral_radius, leak_rate, and input_scaling must be finite; output_scale must be finite and > 0; a NaN from a corrupt file or an unset field lands here |
| Throw on Encode | The field must be FieldSize() long and dst CodeSize() long |
| Throw on EncodeAction | Both the picture and dst must be CodeSize() long; the action cube is k, not dim |
| Throw on Rollout | WorldModel: actions must be a whole number of codes, and out one code longer than that. VectorModel: actions are H × act_dim; last-dim must match act_dim once that is set; the error names both numbers |
| SetHead throws | The Head must be fitted |
| Head::ScoreOn has no AUC | AUC is only computed when y is strictly 0/1 with both classes present; ties count 0.5 |
| VectorModel::Load | A bare `HWM1` constructs a wm-only VectorModel. WorldModel::Load of an `HVM1` file throws bad magic |
| Throw on PaintStripes | The source must be non-empty and no longer than the destination |
| Loss looks huge | It is 0.5 × **sum** of squared error over the code, not a mean |
| Loss never falls | Check the cycle order, and that BeginBatch runs per batch, not per epoch |
| Two actions give the same prediction | The Predictor may be ignoring E(a); raise the action encoder's output_scale, and check that one view code with different action codes gives different predictions |
| A code changed between runs | It cannot; the encoders are frozen. Check the field, or the config seeds |
| Learning rate never decays | Call SetEpoch(epoch, num_epochs) each epoch; an lr_min_frac of one also means constant |
| Second run behaves oddly | ResetTraining; Adam moments and step count persist |
| RestoreBest did nothing | restore_best must be true, and Observe must have seen a finite, lower metric |
| Weights load rejected | LoadWeights needs the exact current weight count: same k, z_max, and gather_span |
| Load throws | The file carries the config and checks the weight count; a hand-edited file will not pass |
| Decoder::Create throws on k | It must be strictly less than dim; pair it with the WorldModel's k |
| Decode output is nonsense | FitInputScale was skipped, or was fitted on other codes; it is one constant and it is saved with the weights |
| FitInputScale throws | The codes were empty, all zero, or held a non-finite value |

## Model persistence

Save writes the config and the Predictor weights. Load reads them,
rebuilds both encoders from the seeds in the config, and puts the
weights back. Encode, EncodeAction, and Predict on the reloaded
instance reproduce the original exactly; the quick start checks all
three on a fresh transition. The Python package reads and writes the
same file.

A passes of 0 means a full tour of whichever cube the encoder sits
on: the view encoder runs N passes, the action encoder 2ᵏ. Config()
keeps passes as given (0 stays 0), so feeding Config() back to Create
rebuilds the same pair. Save writes that same value. RequestedPasses()
is Config().encoder.passes.

| Mechanism | What is stored | Optimizer state? |
|-----------|----------------|------------------|
| WorldModel Save / Load | Config, Predictor weights (`HWM1`) | **No** |
| WorldModel Weights() / LoadWeights() | Predictor weights, verbatim | **No** |
| Decoder Save / Load | Config, input scale, weights | **No** |
| VectorModel Save / Load | New magic `HVM1` (file version 2): WorldModel payload plus optional norms, LCN Heads, dims, bounds, host metadata | **No** |

VectorModel::Load of a bare `HWM1` file constructs a wm-only
VectorModel. WorldModel::Load of an `HVM1` file throws bad magic. Do
not stuff norms or heads into `HWM1`.

Adam moments, step count, and the best snapshot are not saved. To
continue training after Load, the schedule starts cold.

## Limitations

- One WorldModel is **not thread-safe** for concurrent public calls,
  with one exception: Pack is const, writes only into the buffer you
  hand it, and may be called from any number of threads at once. The
  same goes for the size and config getters. Encode, EncodeAction,
  Predict, Rollout, and the training calls are exclusive to one thread
  of control. A Decoder is exclusive to one thread in the same way.
  A Head is exclusive too: Predict, ScoreOn, and PlanCost write the
  LCN's forward state even though Predict is const.
- **Replicas.** Do not clone the WorldModel per thread; that copies
  both frozen encoders for nothing. Keep one WorldModel, encode on it,
  and give each thread its own Predictor built from the WorldModel's
  config: dim k+1, and the same z_max, gather_span, tanh_last, seed,
  and training. Each thread Packs through the shared WorldModel into
  its own buffer and calls Predict(z, hat) or Accumulate on its Predictor. To
  train, each thread first LoadWeights from the WorldModel's Weights,
  then BeginBatch and Accumulate its share; afterwards the master does
  BeginBatch, AddGrad with each replica's Grad, and EndBatch. Predictor
  is in Predictor.h. ThreadPool (ThreadPool.h) is a persistent worker
  set if you want one; it is not required.
- Predict and Rollout are one forward pass per step, single-threaded.
  A sampling planner issues many thousands of them per environment
  step. Batched prediction inside the library is the obvious next
  addition once there is a planner to measure it against.
- EncodeAction is not an embedding lookup. It is a full episode on
  the k-cube, T passes, and it costs more than a Predict. A sampler
  that paints and encodes every raw action it draws pays that episode
  for every sample at every horizon step, and that, not Predict, is
  the hot path. Encode each distinct action once and reuse its code;
  a planner that samples from a finite set of actions encodes the set
  up front and never calls EncodeAction inside the loop.
- Neither class is copyable or movable; Create and Load hand you a
  unique_ptr.
- A Predict is one Predictor pass: z_max × 2ᵏ⁺¹ × (k+1) × gather_span
  multiply-adds plus a tanh per vertex per depth. Accumulate is
  roughly three times that.
- An encoder step reads, for every vertex, its dim neighbours in the
  input field and its dim neighbours in each of the M delay-line
  slices: dim × (M+1) multiply-adds and one tanh per vertex. A view
  episode is T such steps over N vertices; an action episode is T
  steps over 2ᵏ vertices with k in place of dim.
- Memory: Predictor weights are 2ᵏ⁺¹ × (k+1) × gather_span × z_max
  floats, plus four buffers of that size for training. Each encoder
  holds its weights and a delay line of M cubes. A Decoder is an LCN
  on the full dim-cube: N × dim × gather_span × z_max weights, plus
  the same four training buffers, so it is the largest thing in the
  build when dim is large.
- The planner itself (CEM, MPPI, …) is a host choice. VectorModel is
  the model surface a planner talks to, not a planner. RankMe (SVD)
  stays out of this stdlib-only core.

## Dependencies

| Layer | What |
|-------|------|
| Library | C++ standard library only |
| Build | C++23 compiler, CMake 3.21 or later |

There are no third-party dependencies.
