# HypercubeWorldModel Python SDK

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| WorldModel | The product class: two frozen Encoders plus a trained Predictor. |
| Decoder | Reconstruction: a code of 2ᵏ values in, a field of N out. Separate class. |
| Encoder | Frozen hypercube reservoir. View encoder on a dim-cube; action encoder on a k-cube. |
| Predictor | Locally connected net that maps E(x) and E(a) to the next E(x). The only part that trains. |
| dim | View cube dimension. N = 2ᵈⁱᵐ. |
| N | View field length, 2ᵈⁱᵐ. WorldModel.N. |
| k | Code face dimension and action cube dimension. code_size = 2ᵏ. |
| code, E(x), E(a) | Length-2ᵏ arrays. E(x) is the view code; E(a) is the action code. |
| field | N values, one per view-cube vertex. What encode reads. |
| paint_stripes | Lay a short vector onto a field as contiguous stripes. |
| code_size, latent_dim | 2ᵏ. A planner adapter may expose this as latent_dim. |
| passes | Episode length T. 0 means a full tour of that encoder's cube. |
| z_max | Predictor depth. 0 means k+1. |
| output_scale | Encoder presentation gain at Create. Default 1. View and action can be set independently after. |
| rollout | WorldModel.rollout takes action **codes**. A vector adapter's rollout takes raw actions and encodes them once. |
| action_space | Bounds: .low and .high. Not a gymnasium Box. The WorldModel never sees bounds. |

HypercubeWorldModel is a **world model with frozen encoders** on a
Boolean hypercube. Its neurons sit on the vertices of the cube: a cube
of dimension dim has N = 2ᵈⁱᵐ of them. A view, meaning one
observation, comes in as a field of N values, one per vertex. A frozen
hypercube reservoir, the Encoder, runs an episode on that field and
returns a cube of N values, and the first 2ᵏ of them are kept, for
some k smaller than dim. Those 2ᵏ values are the view's code, written
E(x). It is what other systems call the latent. An action, meaning
what the agent did, comes in as a picture of 2ᵏ values and goes
through a second frozen Encoder on a cube of dimension k; its whole
output is the action code, E(a). A locally connected net, the
Predictor, learns to map E(x) and E(a) to the next E(x). It is the only
part of the world model that trains.

Two classes own the product. **WorldModel** is the world model: the
two encoders and the Predictor. **Decoder** is reconstruction: a
second locally connected net that takes a code of 2ᵏ values and gives
back a field of N, trained on its own. One function, **paint_stripes**,
turns a short vector into a field. A planner needs only the first
class. A task that has to see what a code stands for needs the second.

This is a **map API**, not a stream API: one field in, one episode, one
code out. The model does not remember the last view. State that has to
persist across steps lives with the caller.

C++ core and contracts: **[CPP_SDK.md](CPP_SDK.md)**.  
The components, one by one: [encoder.md](encoder.md) / [predictor.md](predictor.md) / [world_model.md](world_model.md) / [decoder.md](decoder.md).  
PyPI-facing package story: **[python/README.md](../python/README.md)**.  
Package version: single source python/hypercube_worldmodel/_version.py.
The wheel metadata, hypercube_worldmodel.__version__, and the compiled
core all read it, and the core refuses to build if it disagrees with
WorldModel.h.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What a step is](#what-a-step-is)
- [API reference](#api-reference)
- [The Decoder](#the-decoder)
- [Shapes](#shapes)
- [Driving the model from a planner](#driving-the-model-from-a-planner)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Installation

### From PyPI (preferred)

Pre-built **wheels**, no compiler required:

```bash
pip install hypercube-worldmodel
```

Import as hypercube_worldmodel (PyPI name hypercube-worldmodel).
Wheels cover Python 3.10 through 3.14 on common Windows (x64), Linux
(x86_64, aarch64), and macOS (x86_64, arm64) builds. NumPy is the only
runtime dependency.

### From source (full repository)

Compile only from a **full clone** of HypercubeWorldModel. The extension
compiles the C++ core, which sits **outside** the python package
directory; a python-only tree is not enough.

Requirements: Python 3.10 or later, a C++23 compiler (GCC 13+, Clang
17+, MSVC 2022+), CMake 3.21 or later, scikit-build-core, pybind11,
NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeWorldModel.git
cd HypercubeWorldModel/python
pip install .
```

On Windows with MinGW (for example the CLion toolchain):

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

From the repository root after install:

```bash
pip install "./python[test]"
pytest python/tests -v --import-mode=importlib
```

Importlib mode avoids the source tree shadowing the installed
extension. Run from the repository root, not from inside python: from
there the source package, which has no compiled core, shadows the
installed one.

### Examples

The [Quick start](#quick-start) below is enough after pip install.
Longer demos live in the **git tree** under
[python/examples/](../python/examples/README.md); they are **not** part
of the wheel. From a clone, repository root:

```bash
pip install hypercube-worldmodel   # or: pip install ./python
python python/examples/plane_point.py
python python/examples/plan_toy.py
```

## Quick start

A point on a plane moves by a bounded velocity. The view is the
position, the action is the velocity.

```python
import numpy as np
import hypercube_worldmodel as hw

rng = np.random.default_rng(0)
count = 512
obs = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
act = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
obs_next = np.clip(obs + 0.1 * act, -1, 1)

wm = hw.WorldModel(dim=6, k=5, passes=12, leak_rate=0.25, input_scaling=0.8,
                   z_max=15, gather_span=5, tanh_last=True,
                   lr=0.03, lr_min_frac=0.05, restore_best=True)

z = wm.encode(hw.paint_stripes(obs, wm.N))                # (count, code_size)
za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
z_next = wm.encode(hw.paint_stripes(obs_next, wm.N))

wm.fit(z, za, z_next, epochs=200, batch_size=16, verbose=True)

hat = wm.predict(z[0], za[0])          # (code_size,) predicted next view code
path = wm.rollout(z[0], za[:3])        # (4, code_size): z[0] then three steps
wm.save("model.wm")
```

### Explicit (full control)

fit is nothing but this loop. Drive it yourself to interleave your own
metrics, schedules, or early stopping:

```python
for epoch in range(epochs):
    wm.set_epoch(epoch, epochs)               # cosine learning rate
    for start in range(0, count, batch_size):
        wm.begin_batch()
        idx = slice(start, start + batch_size)
        wm.accumulate(z[idx], za[idx], z_next[idx])   # forward, loss, backward per row
        wm.end_batch()                        # one Adam step per batch
    wm.observe(wm.evaluate(z_val, za_val, z_val_next), epoch)   # lower wins
wm.restore_best()
```

The order matters and nothing enforces it for you: accumulate reads
the codes you pass and sums into the gradient, end_batch steps the
weights, and the gradient is a **sum** over the batch, so the effective
step scales with batch size.

### The cycle

```text
construct WorldModel
encode every view and encode_action every action once, into your arrays
for each epoch:
    set_epoch                    (cosine learning rate)
    for each batch:
        begin_batch
        accumulate(z, za, z_next)
        end_batch
    evaluate the validation triples
    observe(metric, epoch)
restore_best
predict / rollout / save
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
- The Predictor's cube is k+1, twice the code. Only the first 2ᵏ
  outputs are read; the loss is taken there and nowhere else.
- Nothing in the loss mentions the action. Whether the net reads it is
  a question for a test: feed one view code with two different action
  codes and see whether the prediction changes.
- The Decoder stands outside this diagram. It takes any view code, the
  encoder's or the Predictor's, and gives back a field of N values.

## API reference

### Constructor WorldModel(dim, k, **kwargs)

All knobs are fixed at construction, the same contract as the C++
WorldModelConfig with its EncoderConfig and Predictor knobs laid flat.
Every keyword has a default; the two positional arguments do not.

```python
import hypercube_worldmodel as hw

wm = hw.WorldModel(
    dim,                 # view cube; N = 2**dim; 5 to 24, and at least 6 here
    k,                   # code face and action cube; at least 5, less than dim
    output_scale=...,    # presentation gain on both encoders at Create; finite, > 0; default 1
    encoder_seed=...,    # encoder weight draw (both encoders)
    ic_seed=...,         # encoder episode start state (both encoders)
    spectral_radius=..., # target for the recurrent block; finite, > 0
    leak_rate=...,       # leaky integrator mix; finite, in (0, 1]
    input_scaling=...,   # input drive strength; finite
    history_depth=...,   # delay line length M; 1 to 64
    passes=...,          # passes per episode T; 0 = a full tour of each cube
    z_max=...,           # Predictor depth; 0 = k+1, else >= 2
    gather_span=...,     # Predictor lookback window width; 2 to 6
    tanh_last=...,       # True: tanh on the Predictor's last depth too
    seed=...,            # Predictor weight draw
    lr=..., lr_min_frac=..., lr_decay_epochs=...,   # Adam and cosine schedule
    restore_best=..., beta1=..., beta2=..., eps=...,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| dim | int | View cube dimension. N = 2ᵈⁱᵐ. The Encoder accepts 5 to 24; a WorldModel needs at least 6 so that k has room. |
| k | int | Code face dimension and action cube dimension. At least 5, strictly less than dim. code_size = 2ᵏ. |
| output_scale | float | Presentation gain on both encoders at Create. Finite, > 0. Default 1. Set view and action independently after. |
| encoder_seed, ic_seed | int | Weight draw and episode start state. Both encoders use both; the action encoder differs only in its cube. |
| spectral_radius | float | Target for the recurrent block. Finite, > 0. |
| leak_rate | float | Leaky integrator mix. Finite, in (0, 1]; 1 is full replacement each step. |
| input_scaling | float | Input drive strength. Finite. |
| history_depth | int | Delay line length M, 1 to 64. |
| passes | int | Passes per episode T. 0 means a full tour of whichever cube the encoder sits on: N for the view, 2ᵏ for the action. |
| z_max | int | Predictor depth. 0 means k+1, antipodal reach on the Predictor cube; else at least 2. |
| gather_span | int | Predictor lookback window width, 2 to 6. |
| tanh_last | bool | True applies tanh on the Predictor's last depth too. |
| seed | int | Predictor weight draw. |
| lr | float | Adam step size and cosine peak. Finite, > 0. |
| lr_min_frac | float | Cosine floor as a fraction of lr, in [0, 1]; 1 is a constant rate. |
| lr_decay_epochs | int | Cosine horizon; 0 uses the epochs given to fit or set_epoch. |
| restore_best | bool | Snapshot the Predictor weights on a new low observed metric; fit restores at the end. |
| beta1, beta2 | float | Adam moment decays, in [0, 1). |
| eps | float | Adam denominator floor. Finite, > 0. |

What the encoder knobs do to an episode is in [encoder.md](encoder.md).
What the training knobs do to a run is in [predictor.md](predictor.md).

### Methods

Every method that takes codes or fields accepts one row (a 1-D array)
or many (a 2-D array with one row per sample) and returns the same
shape. The loop over rows runs in C++ with the GIL released.

| Method | Role |
|--------|------|
| encode(fields) | View codes. (N,) or (count, N) in; (code_size,) or (count, code_size) out. |
| last_cube() | Scaled full view episode behind the most recent encode. |
| last_raw_cube() | Unscaled full view episode behind the most recent encode. |
| last_packed() | Packed E(x) then E(a) from the most recent predict or accumulate. |
| encode_action(pictures) | Action codes. (code_size,) or (count, code_size) in and out. A full episode on the action cube per row, not a lookup. |
| predict(z, za) | Predicted next view codes. Matching rows, or one za broadcast against many z. |
| rollout(z0, actions) | Chain predict over a plan of action codes: (H, code_size) or (count, H, code_size) in; (H + 1, code_size) or (count, H + 1, code_size) out, row 0 being z0. |
| pack(z, za) | What the Predictor sees, 2 × code_size per row. Rarely needed. |
| fit(z, za, z_next, *, epochs, batch_size, val, shuffle_seed, verbose) | Shuffle, batch, cosine schedule, restore-best. With val as a (z, za, z_next) tuple the observed metric is evaluate on it, else the mean training loss. Returns self. Calling again continues from the current weights. |
| evaluate(z, za, z_next) | Mean squared error per code value of predict against z_next. Lower is better. |
| begin_batch() | Clear the accumulated gradient. |
| accumulate(z, za, z_next) | Forward, loss, backward for every row; returns the summed loss, 0.5 × SSE over the code per pair. |
| end_batch() | One Adam step on the accumulated gradient. |
| set_epoch(epoch, num_epochs) | Apply the cosine schedule for this epoch. |
| observe(metric, epoch) | Restore-best bookkeeping; **lower wins, strictly**. |
| restore_best() | Write the best-metric snapshot back into the Predictor. |
| reset_training() | Forget the optimizer run: Adam moments, step count, lr, best snapshot. Weights untouched. |
| add_grad(g) | Sum a gradient of the same layout onto the accumulated one, for hosts that train replicas. |
| save(path) / WorldModel.load(path) | The binary file the C++ WorldModel::Save and Load use. |

### Properties

| Property | Meaning |
|----------|---------|
| dim, k | Geometry as given |
| N, code_size | 2ᵈⁱᵐ and 2ᵏ |
| passes, action_passes | Resolved passes per view episode and per action episode |
| z_max | Resolved Predictor depth (0 already replaced by k+1) |
| view_output_scale, action_output_scale | Encoder presentation gains in effect |
| num_weights | Predictor weights: 2ᵏ⁺¹ × (k+1) × gather_span × z_max |
| weights | The Predictor weights as a float32 array, layout depth, axis, tap, vertex. **Settable**, exact length required |
| grad | The accumulated gradient, same layout as weights |
| realized_spectral_radius, action_realized_spectral_radius | The estimate each encoder settled on |

### mean_abs(x) / rms(x)

Mean of absolute values, and root-mean-square, of a float array.
Empty is 0. Use these on a field, `last_raw_cube()`, the k-face
(first `code_size` of that cube), `last_cube()`, or the two halves
of `last_packed()`.

### paint_stripes(x, size)

Lays a short vector onto a field as contiguous stripes: cell j of the
result takes value j × d // size of x, where d is the vector's
length, so each value fills a block of about size / d cells and every
cell is written. x is (d,) or (count, d) with 1 ≤ d ≤ size; the result
is (size,) or (count, size), float32. Use size N for a view and
code_size for an action. It is the same arithmetic as the C++
PaintStripes, and the test suite checks that.

## The Decoder

The Decoder is a locally connected net on the dim-cube that takes a
code of 2ᵏ values, places it on the first 2ᵏ vertices with every other
vertex fed zero, and writes a field of N values: the reconstruction.
Training pairs are a code and the field it came from. Once trained, it
decodes any code of 2ᵏ values, whether encode produced it or predict
did. It is a separate class with its own weights, its own training
cycle, and its own file, and it never touches the WorldModel.

Pairing it with a WorldModel is two numbers: its dim is the
WorldModel's dim and its k is the WorldModel's k.

```python
dec = hw.Decoder(
    dim, k,              # the WorldModel's dim and k; k strictly less than dim
    z_max=...,           # depth; 0 = dim, else >= 2
    gather_span=...,     # lookback window width; 2 to 6
    tanh_last=...,       # True: tanh on the last depth too
    seed=...,            # weight draw
    lr=..., lr_min_frac=..., lr_decay_epochs=...,   # Adam and cosine schedule
    restore_best=..., beta1=..., beta2=..., eps=...,
)

dec.fit(z, fields, epochs=..., batch_size=...)   # fits the input scale first
field = dec.decode(z[0])                          # (N,)
fields = dec.decode(z)                            # (count, N)
dec.save("model.dec")
again = hw.Decoder.load("model.dec")
```

| Method or property | Role |
|--------------------|------|
| decode(codes) | Reconstructed fields. (code_size,) or (count, code_size) in; (N,) or (count, N) out. |
| fit(codes, fields, *, epochs, batch_size, val, shuffle_seed, verbose, fit_input_scale) | The standard cycle. Sets the input scale from codes first unless fit_input_scale is False. val is a (codes, fields) tuple. |
| evaluate(codes, fields) | Mean squared error per field value of decode against fields. |
| fit_input_scale(codes) | Set the input scale to 1 / max abs over every value, so scaled codes lie in [−1, 1]. Frozen after that and saved with the weights. |
| input_scale | The scale in effect. Settable. |
| begin_batch, accumulate(codes, fields), end_batch, set_epoch, observe, restore_best, reset_training, add_grad | The same cycle as WorldModel. accumulate returns the summed loss, 0.5 × SSE over the field per pair. |
| weights, grad, num_weights | As for WorldModel; num_weights is N × dim × gather_span × z_max |
| dim, k, N, code_size, z_max | Geometry; z_max resolved |
| save(path) / Decoder.load(path) | The binary file the C++ Decoder::Save and Load use: config, input scale, weights. |

What the knobs do, and how the reconstruction error behaves as k moves,
is in [decoder.md](decoder.md) and [compression_test.md](compression_test.md).

## Shapes

| Array | Shape | Notes |
|-------|-------|-------|
| a view field | (N,) or (count, N) | one value per vertex of the view cube |
| an action picture | (code_size,) or (count, code_size) | one value per vertex of the action cube |
| a code | (code_size,) or (count, code_size) | what encode, encode_action, and predict return; what predict, accumulate, and decode take |
| a plan | (H, code_size) or (count, H, code_size) | action codes for rollout |
| a rollout | (H + 1, code_size) or (count, H + 1, code_size) | row 0 is the start code |
| a short vector | (d,) or (count, d) | what paint_stripes takes, d ≤ the target size |

Every array is converted to contiguous float32 on the way in; prefer
handing over float32 to avoid the copy. Returned arrays are float32
copies that you own.

## Driving the model from a planner

A sampling planner needs a small surface from a model: encode a view,
encode an action, step a code, roll a code out over a plan, and score
the result. The package gives the first four and leaves the fifth to
the caller, because scoring depends on the task and not on the model.

A sampling planner maps onto the package through a thin adapter.
WorldModel.rollout takes action **codes**. The adapter's rollout
takes **raw** actions (B, H, act_dim), paints and encode_action's
that block once, then calls WorldModel.rollout. CEM is then
copy-paste: it samples in action space and passes those arrays to
rollout.

This path is **state-based**. Concatenate a host observation into a
vector, paint_stripes onto N, encode. A 64×64 RGB frame is
thousands of pixels; N is 2ᵈⁱᵐ (64 at dim 6). paint_stripes cannot
put a camera frame on the cube. Pixels need a painter the host
writes.

| Planner needs | Where | Note |
|---------------|-------|------|
| latent_dim | code_size | 2ᵏ |
| encode(obs) | paint_stripes then encode | obs is (count, obs_dim) **state**. Not a pixel frame |
| encode_action(a) | paint_stripes then encode_action | raw bounded actions. A full action episode per row |
| predict(z, za) | predict | codes in, codes out; batches in C++ |
| rollout(z0, actions) | adapter: raw (B, H, act_dim) | encodes the block once, then WorldModel.rollout on codes |
| action_space.low / .high | adapter.action_space | planner clips here; the WorldModel never sees bounds |
| cost(zs, goal) | caller | distance to a goal code, or a head on frozen codes |

A goal is a view like any other: paint it, encode it, and compare
codes. [python/examples/plan_toy.py](../python/examples/plan_toy.py)
is that adapter, with CEM. For another state-based task, swap the
environment and keep the adapter.

## Error handling

Python-side checks raise ValueError with a short message: a bad array
shape, a code or field of the wrong length, mismatched row counts, a
weight array of the wrong length, epochs or batch_size below 1. The C++
constructors validate every config range and every batched call
validates lengths; their std::invalid_argument maps to ValueError.
File trouble in save and load maps to RuntimeError.

Typical mistakes:

| Symptom | Fix |
|---------|-----|
| ValueError on k at construction | k must be at least 5 and strictly less than dim, so dim must be at least 6 |
| ValueError on a float knob | spectral_radius, leak_rate, input_scaling, output_scale, lr, and eps must be finite; a NaN is rejected even under fast-math |
| ValueError on encode | The field must be N long; a state vector goes through paint_stripes first |
| ValueError on encode_action | The picture must be code_size long, not N; the action cube is k, not dim |
| ValueError on rollout | actions must be (H, code_size) for one start code, (count, H, code_size) for many, with count matching z0 |
| Loss looks huge | accumulate returns 0.5 × the **sum** of squared error over the code, summed over the rows |
| Loss never falls | Check the cycle order, and that begin_batch runs per batch, not per epoch |
| Two actions give the same prediction | The Predictor may be ignoring E(a); raise the action encoder's output_scale, and check that one view code with different action codes gives different predictions |
| A code changed between runs | It cannot; the encoders are frozen. Check the field, or the seeds |
| Second fit behaves oddly | reset_training first; Adam moments and step count persist |
| restore_best did nothing | restore_best must be True, and observe must have seen a finite, lower metric |
| decode output is nonsense | fit_input_scale was skipped, or was fitted on other codes; it is one constant and it is saved with the weights |
| Weights load rejected | The length must equal num_weights: same k, z_max, and gather_span |

## Model persistence

| Mechanism | What is stored | Optimizer state? |
|-----------|----------------|------------------|
| WorldModel save / load | Config, Predictor weights, in the C++ WorldModel file format | **No** |
| Decoder save / load | Config, input scale, weights, in the C++ Decoder file format | **No** |
| pickle, either class | Constructor keywords plus weights (plus the input scale for a Decoder) | **No** |

save writes the same binary file the C++ class writes, so a model
trained in Python loads in C++ and the other way round. load rebuilds
both encoders from the seeds in the file and puts the weights back;
encode, encode_action, and predict on the loaded instance reproduce
the original exactly, and the test suite checks all three with passes
left at 0, the case where the two encoders resolve different T.

pickle captures the constructor keywords as given, passes at 0
included, and the weight array, so an unpickled model rebuilds the
same way. The pickle version is bumped when the layout changes; newer
libraries reject unknown future versions with an upgrade message.

The raw weight array is also yours through the weights property, so
any external format works without either.

**Security:** loading a pickle uses pickle.load. Never unpickle
untrusted files. The save and load files are plain binary with a magic
and a version and carry no code.

## Limitations

- One WorldModel or Decoder is **not thread-safe** for concurrent calls
  from multiple host threads. Separate instances on separate threads
  are fine; the extension releases the GIL during encode,
  encode_action, predict, rollout, decode, accumulate, and end_batch,
  so multi-instance threading gets real parallelism.
- encode_action is a full episode on the action cube, T passes, and it
  costs more than a predict. A sampler that paints and encodes every
  raw action it draws pays that for every sample at every horizon step,
  and that, not predict, is the hot path. Encode each distinct action
  once and reuse the code; a planner with a finite action set encodes
  the set up front.
- predict and rollout loop over rows in C++ but run one Predictor pass
  per row on one thread. A planner that wants a batch across cores
  creates one WorldModel per thread with the same keywords, shares
  weights through the weights property, and drives them from Python
  threads.
- fit drops a short tail each epoch so every Adam step sees exactly
  batch_size samples, unless the whole set is smaller than one batch.
- Native contracts and the C++ surface: **[CPP_SDK.md](CPP_SDK.md)**.

## Dependencies

| Layer | What |
|-------|------|
| Runtime | NumPy |
| Wheel install | No compiler |
| From-source build | Full repo clone, C++23, CMake 3.21 or later, scikit-build-core, pybind11 |

The C++ core is compiled into the extension itself.
