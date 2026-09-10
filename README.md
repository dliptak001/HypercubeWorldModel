# Hypercube World Model

[![Build wheels](https://github.com/dliptak001/HypercubeWorldModel/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeWorldModel/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-worldmodel)](https://pypi.org/project/hypercube-worldmodel/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-worldmodel)](https://pypi.org/project/hypercube-worldmodel/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

HypercubeWorldModel learns how the world moves. It is built from
three core classes.

The **WorldModel** class wraps the other two and manages encoding,
training, and prediction.

The other two form a pipeline: encoder → predictor.

The **Encoder** class is a frozen hypercube reservoir. Its neurons
are the vertices of a Boolean hypercube, one neuron per vertex, and
a cube of dimension dim has N = 2ᵈⁱᵐ of them. What it consumes is a
field: one value per vertex, N values in all. What it returns after
an episode of T passes is a cube: the value every neuron settled on,
again N values. The WorldModel owns two of these encoders, and they
sit on cubes of different sizes.

The **view encoder** sits on the dim-cube. The view is what the world
model sees at one moment: a field of N values, one per vertex. The
episode returns a cube of N values, and only the first 2ᵏ of them are
kept, for some k smaller than dim. Those first 2ᵏ vertices form a
face of the cube, the k-face, and the kept values are the view's
code, written E(x). That cut is the compression: E(x) is 2ᵏ long.

Two things could be meant by compression, and it is worth separating
them now. **Dimensional compression** is the count: N values in, 2ᵏ
out, with the ratio fixed by k and nothing else. That is what this
project does, on purpose and exactly. **Semantic compression** is
whether those 2ᵏ values say something about the whole field rather
than about 2ᵏ vertices of it. The reservoir mixes every vertex with
every other over T passes before the cut, so the k-face carries the
rest of the field with it, and the Decoder can get part of it back.
That carrying is the whole reason the encoder works: a bare slice of
2ᵏ samples would say nothing about the other N − 2ᵏ, and the orbit is
what makes the k-face a code instead of a crop. What the project does
not yet do is put a number on it or steer it. Nothing in the design
sets how much of the field the k-face holds, nothing measures it
apart from what the Decoder gets back, and no knob turns it up or
down. When this README says compression it means the count. The
semantic side is real, it is the point, but it is still unquantified.

The **action encoder** sits on a cube of dimension k, the size of the
face just kept. The action is what the agent did, painted as a
picture of 2ᵏ values, one per vertex. The episode returns a cube of
2ᵏ values and all of it is kept. Nothing is cut, because the action
cube is already the size of the code. The result is the action's
code, written E(a), and it is 2ᵏ long.

So the view comes in at N = 2ᵈⁱᵐ and the action at 2ᵏ, and both leave
their encoders at 2ᵏ. Two codes of the same length is what lets them
sit side by side on the Predictor.

The **Predictor** class is a locally connected net, an LCN, on a
hypercube of dimension k+1, twice the code length. Every vertex has
its own weights over its neighbors and nothing else. One half of the
cube holds E(x), the other half holds E(a), and the net emits a guess
at the next E(x).

A fourth class, the **Decoder**, is a meter. It learns to rebuild the
field from its k-face so you can see what the compression threw
away. It never sits in the prediction loop.

This is a JEPA-shaped world model with frozen encoders: predict the
next code, not the next samples.

The point of this experiment is to see whether a frozen hypercube
reservoir can serve as the encoder of a world model, so that the
Predictor is the only part that has to be trained. The encoders are
drawn from a seed and never modified. The Predictor is small: one
weight table per vertex, over that vertex's neighbors and nothing
else.

A world model does not predict the next view. It predicts the next
code: a short list of numbers that stands in for the view, small
enough to learn on and rich enough that the view could be rebuilt
from it. The usual name for it is the latent; this page says code.
Many systems learn that code. Here it is not learned at all. It is
the first 2ᵏ vertices of the cube the encoder ran on, read straight
off the reservoir.

The aim, then, is a world model whose encoders cost nothing to
train, whose code is a face of the cube it was born on, and whose
dynamics learn from that code alone.

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeEtalon"><strong>HypercubeEtalon</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCascade"><strong>HypercubeCascade</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeLCN"><strong>HypercubeLCN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWorldModel"><strong>HypercubeWorldModel</strong></a>
</p>

<p align="center">
  📄 Foundational paper:
  <a href="docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
  (D.&nbsp;C.&nbsp;Liptak, 2026)
</p>

HypercubeWorldModel is an experiment in the **HypercubeAI** project — our
quest to systematically re-implement classical neural architectures on a
Boolean hypercube topology instead of Euclidean grids or random graphs. The
central thesis is “topology-native intelligence”: the hypercube’s algebraic
structure (vertex-transitive symmetry, Hamming geometry, bitwise addressing)
can serve as a first-class computational substrate.

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the whole
  reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the code is a face of the cube the encoder
  ran on, and the predictor is a net on that same cube grown by one bit. The
  code never leaves the hypercube it was born on, and the action sits one hop
  from the view.
- **Nested codes from one episode** — faces of a hypercube nest by address
  prefix: the (k−1)-face is the first half of the k-face. So one encoder
  episode yields the code at every compression ratio at once, and choosing k
  is choosing where to cut, not rerunning the encoder. A learned encoder has
  to be retrained for each latent size.

Each product in the family is a different architecture on that same foundation.

---

## The Encoder

HypercubeWTF drives a frozen hypercube reservoir with a still field
and hands the result to a readout. HypercubeLCN trains a locally
connected net directly on a cube field. The world model sits between
them: WTF's reservoir as the eye, LCN's net as the dynamics.

The Encoder is a fork of the HypercubeWTF core: one neuron per
vertex, frozen recurrent weights over the cube's edges, a delay line
holding the last M outputs, tanh activation, and a frozen initial
condition that is reloaded before every episode. Nothing in it is
ever trained.

A field has no next sample. So the Encoder invents a short stretch of
synthetic time: it re-addresses the same fixed field over the cube
for T passes and samples the reservoir once at the end. Geometry and
weights stay put; only the registration of the field moves. The
episode goes something like this.

    Leave the caller's field alone. The drive is built in a scratch
    buffer.

    Reload the reservoir's frozen initial condition.

    LOOP:

        Remap the field by xor with the pass index: vertex v is driven
        by the field value at v xor c.

        Inject that remapping. Step the reservoir: every vertex forms
        the weighted sum of its neighbors and its drive, and writes
        tanh of that sum.

        Increment the pass index.

    GOTO LOOP

    After T passes, the reservoir's live output is the cube.

Every episode starts from the same frozen initial condition, so the
cube depends on the field and nothing else. Both the view and the
action go through an episode like this. They differ in which cube
they run on and in how much of the output is kept.

---

## The view

The view is what the world model sees at one moment: a field of N
values laid onto the dim-cube, vertex i holding sample i. The
WorldModel does not define what those samples are. That is up to
the application.

The view encoder runs one episode on that field and returns a cube
of N values. Only the low k-face, 2ᵏ values, is kept. That cut is the
compression, and it is where the world model decides how much of the
view to carry forward. The kept face is E(x).

---

## The action

An action goes through the same machine as the view. The caller
paints it as a picture of 2ᵏ cells, one per vertex of a second, smaller
cube of dimension k. A second Encoder, built from the same knobs and
seeds as the view encoder but on that k-cube, runs one episode on the
picture. Its whole output is already 2ᵏ long, so nothing is cut: that
output is E(a).

The WorldModel does not define what the picture looks like. That is
up to the application. A task with no action at all can pass a
constant picture.

---

## The Predictor

The Predictor's cube has dimension k+1: twice the k-face. The extra
address bit splits it in half. The first half carries E(x). The
second half carries E(a), scaled by one constant so the two codes
reach the net at a chosen ratio. Vertex i of the view sits one hop
from vertex i of the action, so the net's first depth already sees
both.

The net is an LCN: every vertex keeps its own small weight table over
its neighbors and itself, applied depth after depth, with a short
lookback over earlier depths. Training is backprop through every
depth with Adam.

The net has one output per vertex, 2ᵏ⁺¹ in all, but only the first
2ᵏ are used. Those are the predicted next view code, and they are
what the loss compares against the true next view code. The other
outputs are not read by anything.

---

## The Decoder

As a convenience, the Decoder is provided as a reconstruction
meter. It takes the k-face, places it on the low subcube of a full
cube, feeds zero everywhere else, and learns to emit the original
field. The residual, the reconstruction minus the field, is what the
compression threw away in a form the Decoder could not invert. It is
trained separately, scored separately, and never enters the
prediction loop.

---

## Tests

Each test is one program with its own write-up.

- [CompressionTest](tests/CompressionTest.cpp): how much of a field
  survives the cut to a k-face. Write-up:
  [docs/compression_test.md](docs/compression_test.md).
- [JepaEncoderTest](tests/JepaEncoderTest.cpp): does the k-face damp
  noise relative to content, or copy the field. Write-up:
  [docs/jepa_encoder_test.md](docs/jepa_encoder_test.md).
- [JepaPredictorTest](tests/JepaPredictorTest.cpp): the Predictor on a
  stream with no action. Write-up:
  [docs/jepa_predictor_test.md](docs/jepa_predictor_test.md).
- [WorldModelTest](tests/WorldModelTest.cpp): the full WorldModel on a
  stream with a constant action. Write-up:
  [docs/world_model_test.md](docs/world_model_test.md).
- [TerrainWalkerTest](tests/TerrainWalker/TerrainWalkerTest.cpp): a
  walker on an elevation map, the WorldModel predicting the next crop
  from the current crop and the step taken, and the swap check that
  shows the Predictor reads the action. Write-up:
  [docs/terrain_walker.md](docs/terrain_walker.md).

The component specifications are [docs/encoder.md](docs/encoder.md),
[docs/predictor.md](docs/predictor.md),
[docs/decoder.md](docs/decoder.md), and
[docs/world_model.md](docs/world_model.md).

---

## SDKs

**C++** — two classes. WorldModel is the world model: link WorldModel,
include WorldModel.h. Decoder is reconstruction: link Decoder, include
Decoder.h. The guide is [docs/CPP_SDK.md](docs/CPP_SDK.md); the worked
program is [examples/quick_start.cpp](examples/quick_start.cpp).

**Python** — pip install hypercube-worldmodel, import
hypercube_worldmodel. The wheel compiles this C++ core; same two
classes plus paint_stripes, same files. The guide is
[docs/Python_SDK.md](docs/Python_SDK.md); the package story is
[python/README.md](python/README.md).

```cpp
#include "WorldModel.h"
#include "Decoder.h"

WorldModelConfig cfg;
cfg.encoder.dim = dim;               // view cube
cfg.k = k;                           // kept face, k < dim
auto wm = WorldModel::Create(cfg);

std::vector<float> z(wm->CodeSize()), za(wm->CodeSize());
PaintStripes(obs, field);        // a short state vector onto N cells
wm->Encode(field, z);            // field: N floats. z: 2ᵏ floats.
PaintStripes(act, action);       // a short action vector onto 2ᵏ cells
wm->EncodeAction(action, za);    // action and za: 2ᵏ floats.

wm->BeginBatch();
wm->Accumulate(z, za, z_next);   // one training pair
wm->EndBatch();                  // one Adam step

const float* hat = wm->Predict(z, za);   // predicted next code, 2ᵏ floats
wm->Rollout(z, actions, path);           // H action codes in, H+1 view codes out
wm->Save("model.wm");                    // config and Predictor weights

DecoderConfig dcfg;
dcfg.dim = dim;                          // the encoder's cube
dcfg.k = k;                              // the WorldModel's k
auto dec = Decoder::Create(dcfg);
dec->Accumulate(z, field);               // one training pair: code in, field back
const float* rebuilt = dec->Decode(hat); // N floats from a predicted code
```

```python
import hypercube_worldmodel as hw

wm = hw.WorldModel(dim=dim, k=k)
z = wm.encode(hw.paint_stripes(obs, wm.N))                 # (count, 2ᵏ)
za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
wm.fit(z, za, z_next, epochs=epochs, batch_size=batch)
hat = wm.predict(z[0], za[0])                              # (2ᵏ,)
path = wm.rollout(z[0], za[:H])                            # (H+1, 2ᵏ)

dec = hw.Decoder(dim=dim, k=k)
dec.fit(z, fields, epochs=epochs)
field = dec.decode(hat)                                    # (N,)
```

The build is CMake, C++23, no dependencies beyond the standard library
and a thread library. Release builds use fast-math, matching the rest
of the family.
