# Encoder

**Status: implemented in Encoder/.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Encoder | This component: a frozen hypercube reservoir that encodes one field per episode. |
| dim | Dimension of the cube. Valid range [5, 24]. |
| N | Number of vertices, N = 2ᵈⁱᵐ. Also the length of one field. |
| cube | All N vertices. Each vertex is a neuron, labelled by a dim-bit address. |
| field | One sample: N values, vertex i holding sample i. What you encode. |
| neighbor | The vertex reached by flipping one address bit. Every vertex has dim neighbors. |
| episode | One RunEpisode call: load s₀, drive T passes, return the newest slice. |
| s₀ | Start state: N × M floats of delay-line history, drawn once at construction. State, not weights. |
| ic_seed | Seed for drawing s₀. Separate from the weight seed. |
| M | History depth: how many past outputs the delay line keeps. Valid range [1, 64]. |
| slice | One age of the delay line, N floats. Age 0 is the newest, unscaled; RawCube. RunEpisode returns that slice times output_scale. |
| T | Passes per episode (config field `passes`). 0 means T = N, a full tour. |
| c | Pass counter within an episode, 0 .. T−1. |
| drive | The length-N field injected on one pass: the input re-addressed by XOR with c. |
| leak rate | Mix between the previous output and the new tanh. 1 means full replacement. |
| input scaling | How hard the field drives the cube. Input weights are drawn U(−1, 1), then multiplied by input_scaling / √dim. |
| output_scale | Presentation gain on the cube RunEpisode returns. Finite, > 0. Default 1. Does not enter Step. RawCube is the unscaled delay-line slice. |
| spectral radius | Target size of the recurrent operator, applied to the recurrent weights only. |
| seed | Master seed for the weight draws. Named substreams keep input, recurrent, and probe draws from colliding. |
| k | Dimension of the face kept for the Decoder. Strictly less than dim. Compression is this cut. |

A still field has no next sample. The encoder invents a short stretch
of time: it shows the same field to the cube T times, each time under
a different addressing, and reads the reservoir once at the end. The
weights never move. The start state is the same every episode. The
same field therefore always produces the same cube. That is the
encoding.

The cube it returns has N values. Compression is keeping a k-dimensional
face of that cube, k < dim, and handing only that prefix to the
Decoder. The encoder itself always produces the full cube; the cut is
the caller's. How the Decoder puts a field of length N back from that
face is [decoder.md](decoder.md).

The Encoder lives in Encoder/ and depends on nothing else in this
tree. Nothing in Encoder includes Decoder.

## The cube

Label each vertex with a dim-bit integer. Two vertices are neighbors
exactly when their labels differ in one bit. To walk to a neighbor,
flip that bit:

```
v_nn = v XOR (1 << axis)
```

A field is one float per vertex, laid out in address order. The input
is a field. The output is a field. In between, each vertex looks only
at its neighbors — dim of them, never the whole cube, and never
itself.

## Weights and a step

At construction the encoder draws two blocks of weights and then
freezes them. Nothing in an episode touches them.

The input block is N × dim: each vertex has one weight per neighbor
for the staged drive. Those weights start as U(−1, 1) and are then
scaled by `input_scaling / √dim`. That knob is how hard the field
pushes the cube.

The recurrent block is N × M × dim: each vertex has one weight per
neighbor per delay-line age. Those weights are drawn the same way,
scaled by `1 / √(dim · M)`, then rescaled as a block so the delay-line
operator has spectral radius near the configured target. The target
is a construction setting; `RealizedSpectralRadius()` is the
estimate after that rescale.

One step at a vertex: sum the neighbor drives through the input
weights, sum the neighbor history through the recurrent weights, take
tanh, and mix with the previous output by the leak rate. Leak rate 1
replaces the output entirely. The delay line then ages: the new
output becomes age 0, and the staged drive is cleared.

## An episode

Call RunEpisode once per field. Inside, in order: reload s₀, then for
each of T passes build the re-addressed drive, inject it, and step;
then return a pointer to the newest slice.

**Start state s₀.** N × M floats, one full delay line. Drawn once, at
construction, from `ic_seed`, i.i.d. uniform on [−0.5, 0.5]. This is
state, not weights. Each episode copies it into the live delay line
and sets c to 0, so every field starts the orbit from the same place
and nothing carries over from the field before.

**The drive.** The caller's field x is never modified. On pass c,
vertex v is driven by the value at address `v XOR c`, masked to N−1.
XOR with c is a fixed, invertible re-addressing: the geometry and the
weights stay put, only the registration of the field moves.
Incrementing c and driving again is the synthetic time series — the
same picture, T successive addressings. T = 0 in the config means
T = N, a full tour. Larger T wraps.

**Read once.** After the last pass, RunEpisode returns the newest
slice, N floats, valid until the next RunEpisode.

```
const float* out = enc.RunEpisode(x);   // x: N floats. out: N floats.
```

That pointer is the full encoder state. The Decoder takes the first
2ᵏ of it, k < dim, and nothing else.

## Files

```
Encoder/
    Encoder.h        EncoderConfig, class Encoder
    Encoder.cpp
```

CMake target Encoder, a static library with no further links.

## Interface

```cpp
struct EncoderConfig
{
    size_t   dim;               // cube dimension; N = 2ᵈⁱᵐ; [5, 24]
    uint64_t seed;              // weight draws
    float    spectral_radius;  // recurrent target, > 0
    float    leak_rate;         // (0, 1]
    float    input_scaling;     // drive strength
    size_t   history_depth;     // M; [1, 64]
    size_t   passes;            // T; 0 means T = N
    uint64_t ic_seed;           // s₀ draw; separate from seed
    float    output_scale;      // presentation gain on the returned cube; > 0; default 1
};

class Encoder
{
public:
    static std::unique_ptr<Encoder> Create(const EncoderConfig& cfg);

    // One episode. x is not modified. Newest slice times output_scale,
    // N floats, valid until the next RunEpisode. Delay line stays raw.
    const float* RunEpisode(std::span<const float> x);
    const float* RawCube() const;          // unscaled newest slice
    const float* ScaledCube() const;       // RunEpisode return; raw * output_scale

    float SuggestOutputScale(float target_rms = 1.f) const;
    float SuggestOutputScale(std::span<const float> fields, float target_rms = 1.f);
    void FitOutputScale(float target_rms = 1.f);
    void FitOutputScale(std::span<const float> fields, float target_rms = 1.f);
    float OutputScale() const;
    void SetOutputScale(float scale);

    size_t Size() const;                   // N
    EncoderConfig Config() const;          // passes already resolved
    float RealizedSpectralRadius() const;
};

float MeanAbs(std::span<const float> x);   // 0 if empty
float Rms(std::span<const float> x);       // 0 if empty
float SuggestedOutputScale(std::span<const float> raw, float target_rms = 1.f);
```

Validation at Create: dim in [5, 24], spectral radius finite and
positive, leak rate finite and in (0, 1], input scaling finite,
output_scale finite and > 0, history depth in [1, 64]. The finiteness
test is on the bits, since std::isfinite is unreliable under
fast-math. `passes = 0` is stored as N. RunEpisode throws if x is not
length N. SuggestedOutputScale is target_rms / rms(raw), not 1/max.

## Contract summary

- Weights and s₀ are fixed at Create. An episode does not train
  anything.
- RunEpisode always reloads s₀ first. Episodes do not chain.
- The caller's field is not modified. The drive is built in scratch.
- The pointer from RunEpisode is the newest slice times output_scale,
  N floats, and is invalid after the next RunEpisode. RawCube is the
  unscaled delay-line slice; ScaledCube is the same pointer RunEpisode
  returned. MeanAbs and Rms measure any span.
- Config reports the resolved T, not a stored 0.
- Compression is the k-face the caller keeps. The encoder always
  returns N.
