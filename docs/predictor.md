# Predictor

**Status: implemented in Predictor/.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Predictor | This component: map one k-face to a predicted next k-face. |
| Encoder | The frozen hypercube reservoir in Encoder/, specified in [encoder.md](encoder.md). |
| Decoder | Reconstruction meter in Decoder/. Not part of this map and not used to train it. |
| LCN | Locally connected net on a Boolean hypercube, in LCN/. The Predictor owns one. |
| LCNTraining | The LCN's gradient trainer (backprop through every depth, Adam). The Predictor owns one. |
| k | Dimension of the encoder face kept as the latent. Strictly less than the Encoder's dim. |
| dim | Dimension of the Predictor's own cube. Not the Encoder's dim. Equal to k when the caller already sliced to the k-face; WorldModel uses dim = k+1. |
| N | Number of vertices on this cube, N = 2ᵈⁱᵐ. Length of one Forward. |
| k-face | The low 2ᵏ vertices of an encoder episode. |
| E(x) | Encoder output on field x, sliced to the k-face. |
| xₜ, xₜ₊₁ | Consecutive observation windows of a stream. |
| ŝ | The Predictor's output: Size() floats. WorldModel reads the first 2ᵏ as the next k-face. |
| a | Action field. Not an input here. WorldModel concatenates E(x) and E(a) onto a (k+1)-cube and passes that cube in. |
| first subcube | Vertices whose extra address bit is 0 when dim = k+1. Length 2ᵏ. |
| extra bit-face | Vertices whose extra address bit is 1 when dim = k+1. WorldModel puts E(a) here. |
| prefix | Accumulate may take a next shorter than N. Only that prefix is the target; the rest of the cube is unconstrained. |
| z_max | LCN depth. 0 means dim. dim is one full antipodal reach. |
| gather_span | LCN lookback window width in fields. |
| tanh_last | Whether the LCN's last depth applies tanh. |
| seed | Seed for the LCN's initial weight draw. |
| batch | One BeginBatch, some Accumulate calls, one EndBatch. Gradients sum across the batch. |
| epoch | One pass over the training pairs; SetEpoch sets the learning rate for it. |
| restore_best | Snapshot the weights on a new low metric; put them back at RestoreBest. |

A JEPA world model predicts the next latent, not the next samples.
The Encoder turns a window x into a k-face E(x). The Predictor is
the map P that takes that face and emits a guess at the face of the
next window:

ŝ = P(E(xₜ))

The guess lives on this cube. Action is not an input here. When the
caller is WorldModel, E(a) sits beside E(x) by growing this cube one
bit: first subcube is E(x), extra bit-face is E(a), and loss is the
prefix of length 2ᵏ. See [world_model.md](world_model.md).
The Decoder is a reconstruction meter on the same encoder. It is not
in this loop: training the Predictor does not decode, and scoring it
does not reconstruct the field.

The Predictor lives in Predictor/ beside Encoder/ and LCN/ and
depends only on LCN. It does not include the Encoder. The caller
slices each episode down to 2ᵏ and passes that in.

## Latent cube

The LCN runs on a cube of dimension dim, not on the Encoder's cube.
If the Encoder is dim = 8 and the kept face is k = 7, a caller that
already sliced to the k-face uses Predictor dim = 7 and N = 128.
WorldModel uses dim = k+1 instead and concatenates E(x) and E(a).
Compression already happened when the caller dropped the rest of the
encoder cube. Feeding the Predictor the full encoder state would skip
that cut.

Forward input is always Size() long. The training target may be a
prefix: LCNTraining already allows that, and Accumulate passes it
through. There is no input scale.

## Training

Train on pairs (E(xₜ), E(xₜ₊₁)). Those pairs have to come from a
stream. Independent random fields have no next window; i.i.d. draws
cannot train this map.

One batch is the same cycle the Decoder uses:

```
pred.BeginBatch();
for each pair: pred.Accumulate(z_t, z_next);
pred.EndBatch();
```

Accumulate is Forward, Loss against next, Backward. Loss is 0.5 × SSE
over the targeted prefix (the whole cube when next is Size() long).
SetEpoch, Observe, and RestoreBest are the cosine schedule and
best-weight snapshot, same as LCNTraining. The stale-gradient guards
still apply.

Inference is one Forward:

```
const float* hat = pred.Predict(z);    // z: N floats. hat: N floats.
```

The pointer is the LCN output, valid until the next Predict or
Accumulate.

## Files

```
Predictor/
    Predictor.h        PredictorConfig, class Predictor
    Predictor.cpp
```

CMake target Predictor, a static library linking LCN. Nothing in
Predictor includes Encoder or Decoder.

## Interface

```cpp
struct PredictorConfig
{
    size_t   dim;          // latent cube; typically Encoder k
    size_t   z_max;        // LCN depth
    size_t   gather_span;
    bool     tanh_last;
    uint64_t seed;
    LCNTrainingConfig training;
};

class Predictor
{
public:
    static std::unique_ptr<Predictor> Create(const PredictorConfig& cfg);

    // Inference. z.size() must equal Size(). Newest k-face, N floats,
    // valid until the next Predict or Accumulate.
    const float* Predict(std::span<const float> z);

    // Training cycle, one batch:
    //   BeginBatch(); for each pair: Accumulate(z_t, z_next); EndBatch();
    // Accumulate returns the sample loss (0.5 * SSE over the targeted prefix).
    // next may be shorter than Size(); the rest of the cube is free.
    void  BeginBatch();
    float Accumulate(std::span<const float> z, std::span<const float> next);
    void  EndBatch();

    void  SetEpoch(int epoch, int num_epochs = 0);
    void  Observe(float metric, int epoch);
    void  RestoreBest();
    void  ResetTraining();

    const std::vector<float>& Weights() const;
    void  LoadWeights(std::span<const float> w);
    const std::vector<float>& Grad() const;
    void  AddGrad(std::span<const float> g);  // reduce a replica shard onto this net

    size_t Size() const;                 // N = 2ᵈⁱᵐ
    const PredictorConfig& Config() const;
};
```

Validation at Create is whatever LCN and LCNTraining already check.
Persistence is WorldModel's job (Save / Load of the Predictor weights).
Replicas share weights with LoadWeights.

## What this is not

It is not a reconstruction. The Decoder still answers "does the
k-face still carry the field?" The Predictor answers "does the
k-face of this window tell you the k-face of the next one?" Those
are different meters. E(x) cat E(a) is packed by WorldModel, not by this class.

The public product face that owns Encoder plus Predictor and does
the k-cut is [world_model.md](world_model.md).
