# Hypercube World Model

[![Build wheels](https://github.com/dliptak001/HypercubeWorldModel/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeWorldModel/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-worldmodel)](https://pypi.org/project/hypercube-worldmodel/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-worldmodel)](https://pypi.org/project/hypercube-worldmodel/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeWorldModel/blob/master/LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

This package is the **Python** surface for HypercubeWorldModel
(import hypercube_worldmodel).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeWorldModel](https://github.com/dliptak001/HypercubeWorldModel)**.

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
down. When this page says compression it means the count. The
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

In Python the product is two classes, **hypercube_worldmodel.WorldModel**
and **hypercube_worldmodel.Decoder**, plus one function,
**paint_stripes**, that turns a short vector into a field.

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
  <a href="https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
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
the application. A state vector shorter than N goes through
paint_stripes, which lays each value over a block of the cube.

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
constant picture. A bounded action vector goes through paint_stripes
the same way a state vector does.

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
prediction loop. In Python it is the second class, and it decodes a
predicted code as readily as an encoded one.

---

## Installation

**Preferred:** install a pre-built wheel from PyPI (no compiler).

```bash
pip install hypercube-worldmodel
```

```python
import hypercube_worldmodel as hw
print(hw.__version__)
```

Package name on PyPI: **hypercube-worldmodel**. Import name:
**hypercube_worldmodel**. Main types: **hw.WorldModel** and
**hw.Decoder**.

The wheel compiles the C++ WorldModel and Decoder into the extension.
It is the same net as the C++ SDK, the same files, not a port.

Wheels target Python 3.10 through 3.14 on common Windows, Linux, and
macOS machines. Runtime dependency: NumPy only.

### From source (full repository)

To compile the extension yourself, clone the **entire** repository,
not the python folder alone: the C++ core lives next to it. You need
Python 3.10 or later, a C++23 compiler, and CMake 3.21 or later.

```bash
git clone https://github.com/dliptak001/HypercubeWorldModel.git
cd HypercubeWorldModel/python
pip install .
```

On Windows with CLion's MinGW, put that compiler's bin folder and
Ninja on your PATH, then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

(Exact CLion paths change with the version.)

---

## Quick start

A point on a plane moves by a bounded velocity. The view is the
position, the action is the velocity. Both are short vectors, so they
go through paint_stripes, which spreads each value over a block of the
cube so the encoder has something to respond to.

Shapes that matter:

| Array | Shape | Notes |
|-------|-------|-------|
| a view field | (N,) or (count, N) | one value per vertex of the view cube |
| an action picture | (code_size,) or (count, code_size) | one value per vertex of the action cube |
| a code | (code_size,) or (count, code_size) | what encode, encode_action, and predict return |
| a plan | (H, code_size) or (count, H, code_size) | action codes for rollout |

```python
import numpy as np
import hypercube_worldmodel as hw

rng = np.random.default_rng(0)
count = 512
obs = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
act = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
obs_next = np.clip(obs + 0.1 * act, -1, 1)

wm = hw.WorldModel(dim=6, k=5, passes=12, leak_rate=0.25, input_scaling=0.8,
                   action_scale=0.33, z_max=15, gather_span=5, tanh_last=True,
                   lr=0.03, lr_min_frac=0.05, restore_best=True)

z = wm.encode(hw.paint_stripes(obs, wm.N))                # (count, code_size)
za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
z_next = wm.encode(hw.paint_stripes(obs_next, wm.N))

wm.fit(z, za, z_next, epochs=200, batch_size=16, verbose=True)

hat = wm.predict(z[0], za[0])          # (code_size,) predicted next view code
path = wm.rollout(z[0], za[:3])        # (4, code_size): z[0] then three steps
wm.save("model.wm")                    # the same file C++ WorldModel::Load reads
```

To see what a code stands for, train a Decoder beside it:

```python
dec = hw.Decoder(dim=wm.dim, k=wm.k, gather_span=3, lr=0.01, restore_best=True)
dec.fit(z, hw.paint_stripes(obs, wm.N), epochs=100, batch_size=16)
field = dec.decode(hat)                # (N,) the predicted next view, rebuilt
```

### Step by step (same loop, more control)

fit is nothing but this loop. Drive it yourself to interleave your
own metrics, schedules, or early stopping:

```python
for epoch in range(epochs):
    wm.set_epoch(epoch, epochs)               # cosine learning rate
    for start in range(0, count, batch_size):
        wm.begin_batch()
        idx = slice(start, start + batch_size)
        wm.accumulate(z[idx], za[idx], z_next[idx])   # forward, loss, backward per row
        wm.end_batch()                        # one Adam step per batch
    wm.observe(wm.evaluate(z_val, za_val, z_val_next), epoch)
wm.restore_best()
```

The gradient is a **sum** over the batch, so the effective step scales
with batch size, the same convention as the C++ SDK.

### Driving a planner

A sampling planner needs five calls from a model: encode a view,
encode an action, step a code, roll a code out over a plan, and score
the result. The package gives the first four; scoring is the task's.
[python/examples/plan_toy.py](https://github.com/dliptak001/HypercubeWorldModel/blob/master/python/examples/plan_toy.py)
is the DeepMind Control Suite protocol: CEM samples raw actions,
reads action_space.low / .high, and passes those arrays to rollout.
The adapter paints and encode_action's the block once, then
WorldModel.rollout on the codes. This path is state-based: concatenate
the observation into a vector, paint_stripes onto N, encode. A camera
frame is not a field of N. Swap the environment for a state-based
suite task and keep the adapter.

---

## Features

- **Two classes.** hw.WorldModel and hw.Decoder are the whole surface,
  plus one function, hw.paint_stripes.
- **Frozen encoders.** Codes depend only on the field and the seeds;
  encode a stream once, train on it many times.
- **fit.** Shuffle, batch, cosine schedule, restore-best with an
  optional validation set, in one call.
- **Custom loops.** begin_batch, accumulate, end_batch, set_epoch,
  observe, restore_best, exposed one to one with the C++ API.
- **Batched by shape.** Every method takes one row or many and returns
  the same shape; the loop runs in C++ with the GIL released.
- **rollout.** Chain predict over a plan of action codes, for a
  sampling planner.
- **Nested codes.** Faces nest by address prefix, so the first half of
  a code is the code one k lower; last_cube gives the whole episode.
- **Weights and gradient as NumPy.** Settable weights and a readable
  gradient on both classes, for hosts that train replicas.
- **Save and load.** The same binary files the C++ classes write and
  read, and pickle for convenience. Optimizer state is not stored.
- **NumPy float32.** Arrays converted for you; prefer contiguous
  float32.

---

## Examples

For a first try, paste the [Quick start](#quick-start) after
pip install hypercube-worldmodel. That is self-contained.

The demo scripts on GitHub under
[python/examples/](https://github.com/dliptak001/HypercubeWorldModel/tree/master/python/examples)
are there to open or download; they are not added to your machine by
pip.

| Script | What it is for |
|--------|----------------|
| [plane_point.py](https://github.com/dliptak001/HypercubeWorldModel/blob/master/python/examples/plane_point.py) | The quick start with a held-out score against the identity guess, a Decoder, and a save and load round trip |
| [plan_toy.py](https://github.com/dliptak001/HypercubeWorldModel/blob/master/python/examples/plan_toy.py) | DMC protocol adapter: CEM passes raw actions to rollout, action_space.low / .high; state vectors, not pixels |

```bash
# from a clone of HypercubeWorldModel, after: pip install hypercube-worldmodel
python python/examples/plane_point.py
python python/examples/plan_toy.py
```

These use an easy made-up world so the API is obvious; they are not
scores to publish.

---

## Tests and write-ups

The C++ tests are where the design was worked out, and each one has
its own write-up in the repository.

- [CompressionTest](https://github.com/dliptak001/HypercubeWorldModel/blob/master/tests/CompressionTest.cpp): how much of a field
  survives the cut to a k-face. Write-up:
  [docs/compression_test.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/compression_test.md).
- [JepaEncoderTest](https://github.com/dliptak001/HypercubeWorldModel/blob/master/tests/JepaEncoderTest.cpp): does the k-face damp
  noise relative to content, or copy the field. Write-up:
  [docs/jepa_encoder_test.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/jepa_encoder_test.md).
- [JepaPredictorTest](https://github.com/dliptak001/HypercubeWorldModel/blob/master/tests/JepaPredictorTest.cpp): the Predictor on a
  stream with no action. Write-up:
  [docs/jepa_predictor_test.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/jepa_predictor_test.md).
- [WorldModelTest](https://github.com/dliptak001/HypercubeWorldModel/blob/master/tests/WorldModelTest.cpp): the full WorldModel on a
  stream with a constant action. Write-up:
  [docs/world_model_test.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/world_model_test.md).
- [TerrainWalkerTest](https://github.com/dliptak001/HypercubeWorldModel/blob/master/tests/TerrainWalker/TerrainWalkerTest.cpp): a
  walker on an elevation map, the WorldModel predicting the next crop
  from the current crop and the step taken, and the swap check that
  shows the Predictor reads the action. Write-up:
  [docs/terrain_walker.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/terrain_walker.md).

The component specifications are
[docs/encoder.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/encoder.md),
[docs/predictor.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/predictor.md),
[docs/decoder.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/decoder.md), and
[docs/world_model.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/world_model.md).

The package's own tests are
[python/tests/test_basic.py](https://github.com/dliptak001/HypercubeWorldModel/blob/master/python/tests/test_basic.py):
contract coverage on small cubes, including the save and load round
trip against the C++ file format.

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/Python_SDK.md)** | Canonical Python API: every method, shape, file format, limits |
| [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/CPP_SDK.md) | Native library guide (same product, C++) |
| [Project README](https://github.com/dliptak001/HypercubeWorldModel#readme) | Product story and the C++ SDK from the repo root |
| [docs/encoder.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/encoder.md) | The frozen encoder episode and its knobs |
| [docs/predictor.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/predictor.md) | The Predictor and its training cycle |
| [docs/world_model.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/world_model.md) | The WorldModel class, the action path, persistence |
| [docs/decoder.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/docs/decoder.md) | The Decoder class and its file format |
| [python/examples/README.md](https://github.com/dliptak001/HypercubeWorldModel/blob/master/python/examples/README.md) | The Python demo scripts |

---

## Ecosystem

- **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)**: the frozen reservoir orbit the Encoder is forked from, plus a thin readout.
- **[HypercubeLCN](https://github.com/dliptak001/HypercubeLCN)**: the locally connected net the Predictor and the Decoder are built from, every weight trained.
- **[HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon)**: frozen etalon transit plus a thin readout.
- **[HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)**: both frozen stages in series plus a thin readout.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)**: cube-native conv stack with shared kernels.
- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)**: echo-state reservoir computing on streams.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)**: Hopfield-style dynamics on the cube.

---

## License

Apache 2.0.
