# Decoder

**Status: implemented in Decoder/.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Decoder | This component: invert a compressed encoder state. Subcube in, full cube out. |
| Encoder | The frozen hypercube reservoir in Encoder/, specified in [encoder.md](encoder.md). |
| LCN | Locally connected net on a Boolean hypercube, in LCN/, specified in [lcn.md](lcn.md). The Decoder owns one. |
| LCNTraining | The LCN's gradient trainer (backprop through every depth, Adam). The Decoder owns one. |
| dim | Dimension of the output cube. Same dim as the encoder it is paired with. |
| N | Number of vertices, N = 2ᵈⁱᵐ. Length of the reconstructed field. |
| cube | All N vertices of the LCN. The Decoder's output: the reconstructed field. |
| field | One sample: N values, vertex i holding sample i. What was encoded, and what the Decoder is trained to emit. |
| k | Dimension of the input face. Must be strictly less than dim. Pair with WorldModel's k. |
| code | The Decoder's input: 2ᵏ values, the compressed encoder state. |
| FieldSize | Length of the reconstructed field, N. |
| CodeSize | Length of the input code, 2ᵏ. |
| subcube | The Decoder's input: 2ᵏ values, the compressed encoder state, placed on the LCN's low subcube. |
| low subcube | The vertices whose top address bits are zero: a contiguous prefix of the cube. |
| zero-fed vertices | Vertices of the LCN outside the low subcube. Fed zero on input. They are still outputs: the loss covers the full cube. |
| padded input | The length-N buffer the LCN reads: the scaled subcube followed by zeros. |
| input scale | One constant multiplied into the subcube before placement. |
| residual | The reconstructed field minus the original field. What the compressed code did not give the Decoder in a form it could invert. |
| z_max | LCN depth. 0 means dim. dim is one full antipodal reach. |
| gather_span | LCN lookback window width in fields. |
| tanh_last | Whether the LCN's last depth applies tanh. |
| seed | Seed for the LCN's initial weight draw. |
| batch | One BeginBatch, some Accumulate calls, one EndBatch. Gradients sum across the batch. |
| epoch | One pass over the training set; SetEpoch sets the learning rate for it. |
| restore_best | Snapshot the weights on a new low metric; put them back at RestoreBest. |
| stale-gradient guard | LCNTraining's check that Loss and Backward refer to the same Forward. |
| magic | The four bytes at the start of a saved file that identify it. |

The encoder is a compression engine. After an episode it holds a
full cube of N values; only a k-dimensional face of that cube,
k < dim, is kept. The Decoder's job is to take that face and produce
the full cube again — the reconstructed field. Comparing that
reconstruction to the field that was encoded is how you see what the
compression threw away.

The Decoder lives in Decoder/ beside Encoder/ and LCN/ and depends
only on LCN. It does not include the encoder. The caller slices the
encoder output down to 2ᵏ and passes that in. The episode that
produced the cube is [encoder.md](encoder.md).

## Encode → Decode

```
const float* out = enc.RunEpisode(x);          // x: N floats. out: N floats.
dec.Decode(std::span(out, size_t{1} << k));    // first 2ᵏ only
```

The encoder always produces a full cube. Compression is the slice.
The Decoder refuses k = dim: that would skip the compression and feed
it the full encoder state.

Train the Decoder on pairs `(subcube, x)` — the compressed code, and
the field that produced it. After that, Encode → Decode reconstructs
a new field. The residual is encoder compression and decoder
invertibility together; that is the practical answer, at this stage,
to how much the encoder threw away.

The low subcube is the contiguous prefix of the encoder's newest
slice. Any other choice of fixed bits is the same network under
relabeling for the LCN, but not for the encoder, whose weights are
not symmetric under relabeling. The low face is the one this path
uses.

## Responsibilities

Decoder owns an LCN and an LCNTraining and does one thing: take a
code, produce the full field. The input is always a proper k-face
(`k < dim`).

It handles:

- **Placement.** The input subcube lands on the LCN's own low subcube
  of the same dimension. The remaining vertices are fed zero. They
  are still outputs — every vertex of the reconstructed field has a
  target — so the loss covers the full cube, no mask. The caller
  does not build a padded field.
- **Input scale.** One global scalar applied to the subcube before
  placement. Fitted once from a training set, then frozen and saved
  with the weights. This is the only preprocessing the decoder does.
- **The training cycle.** Batch begin, accumulate (Forward, Loss,
  Backward) per sample, batch end (Adam), epoch schedule, best-weight
  tracking. The LCN and LCNTraining stale-gradient guards still
  apply.
- **Persistence.** Config, input scale, and weights in one file, with
  a magic and version so a mismatched file is rejected.

Depth: each LCN layer is one Hamming hop, and the input lives on a
k-face. A vertex with h high bits set is h hops from that face.
`z_max = dim` is one full antipodal reach and is enough for the
input face to cover the cube.

## Files

```
Decoder/
    Decoder.h        DecoderConfig, class Decoder
    Decoder.cpp
```

CMake target Decoder, a static library linking LCN. Nothing in
Decoder includes Encoder.

## Interface

```cpp
struct DecoderConfig
{
    size_t   dim;          // output cube; same dim as the encoder
    size_t   k;            // input face; must be strictly less than dim
    size_t   z_max;        // LCN depth
    size_t   gather_span;
    bool     tanh_last;
    uint64_t seed;
    LCNTrainingConfig training;
};

class Decoder
{
public:
    static std::unique_ptr<Decoder> Create(const DecoderConfig& cfg);
    static std::unique_ptr<Decoder> Load(const std::filesystem::path& file);
    void Save(const std::filesystem::path& file) const;

    // Inference. code.size() must equal CodeSize(). Newest field,
    // N floats, valid until the next Decode or Accumulate.
    const float* Decode(std::span<const float> subcube);

    // Training cycle, one batch:
    //   BeginBatch(); for each sample: Accumulate(subcube, target); EndBatch();
    // Accumulate returns the sample loss (0.5 * SSE over the full cube).
    void  BeginBatch();
    float Accumulate(std::span<const float> subcube, std::span<const float> target);
    void  EndBatch();

    void  SetEpoch(int epoch, int num_epochs = 0);
    void  Observe(float metric, int epoch);
    void  RestoreBest();
    void  ResetTraining();

    // Input scale: fitted once, frozen, saved. Decode and Accumulate
    // multiply the subcube by it before placement.
    void  FitInputScale(std::span<const float> subcubes);  // concatenated training subcubes
    void  SetInputScale(float scale);

    const std::vector<float>& Weights() const;    // depth, axis, tap, vertex
    void  LoadWeights(std::span<const float> w);
    const std::vector<float>& Grad() const;
    void  AddGrad(std::span<const float> g);
    float InputScale() const;

    size_t CodeSize() const;     // input vertices, 2ᵏ
    size_t FieldSize() const;    // output vertices, N
    const DecoderConfig& Config() const;
};
```

Validation at Create: k must be strictly less than dim,
plus everything LCN and LCNTraining already check. Load rejects a
file whose magic, version, or weight count does not match its own
config.

## File format

Little-endian binary:

```
magic       "HSDC"            4 bytes
version     uint32
config      dim, k, z_max, gather_span, tanh_last, seed
            and the LCNTrainingConfig fields
input_scale float32
n_weights   uint64
weights     float32 * n_weights   (LCN::Weights() order: depth, axis, tap, vertex)
```

The LCN's initial weights are determined by its seed, so config and
scale alone would regenerate an untrained decoder. The weights are
written regardless so a trained file stands on its own.

## Contract summary

- Decode is pure inference. It does not touch gradient state. The
  pointer is the reconstructed field, N floats, invalid after the
  next Decode or Accumulate.
- Accumulate is Forward, Loss, Backward on the underlying LCN and
  LCNTraining, in that order, and returns the sample loss. The
  stale-gradient guard in LCNTraining still applies: an EndBatch
  (Adam) between Accumulate calls of the same batch is a caller error
  and will throw from Backward.
- The target of Accumulate is always the full cube, length FieldSize().
- FitInputScale sets scale = 1 / max absolute value over all values
  passed in, so the scaled input lies in [−1, 1]. It throws if the
  input is empty, all zero, or contains a non-finite value.
- Adam state is not saved. A loaded Decoder starts training fresh.
- Save after RestoreBest is the normal end of a run.
