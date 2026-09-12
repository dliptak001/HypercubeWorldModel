# LCN

**Status: implemented in LCN/.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| LCN | This component: a locally connected net on a Boolean hypercube. One field in, one field out. |
| LCNTraining | The LCN's trainer: backprop through every depth, stepped with Adam. Holds a reference to the LCN. |
| dim | Dimension of this cube. Valid range [4, 24]. Not the Encoder's dim unless the caller is the Decoder. |
| N | Number of vertices, N = 2ᵈⁱᵐ. Length of one Forward. |
| cube | All N vertices. Each vertex is a neuron, labelled by a dim-bit address. |
| field | N values, vertex i holding sample i. Input and output of Forward. |
| neighbor | The vertex reached by flipping one address bit. Every vertex has dim neighbors. |
| depth, z | One layer of the net. Depth z reads a window of past fields and writes the next one. |
| z_max | Number of depths. 0 means dim. dim is one full antipodal reach. Must be at least 2. |
| gather_span | How many past fields each depth looks at. Valid range [2, 6]. |
| tap | One slot in that lookback. Tap k at depth z reads field z+k. |
| tanh_last | Whether the last depth applies tanh. Default false: the last depth writes the raw sum. |
| seed | Seed for the initial weight draw. |
| prefix | Loss may take a target shorter than N. Only that prefix seeds the gradient; the rest of the cube is free. |
| batch | One ZeroGrad, some Forward / Loss / Backward triples, one Adam. Gradients sum across the batch. |
| epoch | One pass over the training set; SetEpoch sets the learning rate for it. |
| restore_best | Snapshot the weights on a new low metric; put them back at RestoreBest. |
| stale-gradient guard | Backward requires an unconsumed Loss against the most recent Forward. |

The Predictor owns this net. It is the only part of the world model
that trains: it guesses the next view code. The Decoder owns one too,
as a reconstruction meter. Head owns one as the planner's scalar.
The Encoder is a frozen reservoir, not an LCN.

A locally connected net on a cube means each vertex keeps its own
small weight table over its neighbors and a short lookback, and
nothing else. There is no dense layer. There is no shared kernel:
vertex v's weights are not vertex u's. The geometry is the
hypercube, so a neighbor is a bit flip and every vertex has the same
number of them.

One Forward takes a field of N values and writes a field of N values.
In between it writes z_max intermediate fields on the same cube.

## The cube

Label each vertex with a dim-bit integer. Two vertices are neighbors
exactly when their labels differ in one bit. To walk to a neighbor,
flip that bit:

```
v_nn = v XOR (1 << axis)
```

A field is one float per vertex, laid out in address order. At each
depth, vertex v looks only at its dim neighbors — never the whole
cube, and never itself in that step. What it sees of itself arrives
later, after a signal has hopped out and hopped back. That is why
z_max must be at least 2.

## A depth

Call Forward once per field. The net keeps a stack of fields:
gather_span slots of lookback, then z_max written depths. The
caller's field is copied into the last lookback slot. The slots
before that stay zero; they are not a delay line that carries from
call to call. Each Forward starts that stack over.

Depth z reads fields z through z+gather_span−1 and writes field
z+gather_span. For each neighbor axis and each tap in the window,
vertex v adds a weight times the neighbor's value. Then tanh, except
the last depth when tanh_last is false.

Each depth is one Hamming hop. A vertex h hops from a signal needs
depth h to see it. z_max = dim is antipodal reach: every vertex can
see every other. Deeper than that is extra capacity, not extra
geometry. If you read only vertex 0, that vertex needs enough depth
to see the parts of the field that matter. z_max = 2 is two hops.
Pooling the output field does not give extra reach: vertices outside
the hop radius are still uninformed.

gather_span is the lookback, not the hop. Span 2 already lets a depth
mix the previous output with the one before it. Wider span is more
history at each hop.

**tanh_last.** Intermediate depths always squash. The last depth, by
default, does not. Squared-error training against values that already
sit near ±1 wants the raw sum: tanh only reaches ±1 asymptotically.
Set tanh_last when the output should itself be a tanh-range code. The
Predictor and the Decoder usually do.

## Weights

At Create the net draws N × dim × gather_span × z_max weights from a
normal with scale 1 / √(dim · gather_span), then trains them. Layout
is depth, axis, tap, vertex, so the vertex loop in Forward and
Backward runs over adjacent floats.

That layout is not HypercubeLCN's. Weights do not move between the
two without a transpose.

LoadWeights replaces the block and invalidates any pending gradient:
the next Backward needs a fresh Forward and Loss.

## Training

LCNTraining is a separate object that holds a reference to the LCN.
The LCN must outlive it. One batch is:

```
train.ZeroGrad();
for each sample:
    net.Forward(x);
    train.Loss(target);     // 0.5 × SSE
    train.Backward();
train.Adam();
```

Loss reads the most recent Forward. The target may be shorter than N:
only that prefix carries a target and a gradient seed; the rest of
the output is unconstrained. The Predictor uses that prefix so
WorldModel can take loss on E(x) and leave E(a)'s half free. A scalar
Head is the same idea with length 1: train vertex 0, read vertex 0.
Do not copy a scalar onto all N vertices. Every targeted vertex adds
to the SSE and to the gradient, so a full-field target is N times a
one-vertex target, and vertices that never saw the signal still try
to match y.

Backward walks the depths in reverse and **sums** into the gradient.
The Adam step therefore scales with batch size, and with how many
vertices the Loss targeted. An lr that is right for a short prefix
is wrong for a full-cube target. The stale-gradient
guard throws if Backward does not see an unconsumed Loss for the
Forward still in the net: Forward(a), Loss, Forward(b), Backward would
otherwise mix b's activations with a's error.

SetEpoch applies a cosine schedule: epoch 0 is the peak, the last
epoch of the horizon is the floor. Observe snapshots the weights on a
new low metric when restore_best is on; RestoreBest writes them back.
Reset forgets Adam, the schedule, and the snapshot; it does not touch
the weights.

There is no Save here. Decoder and WorldModel write the weight block
into their own files.

## Files

```
LCN/
    LCN.h            LCNConfig, class LCN
    LCN.cpp
    LCNTraining.h    LCNTrainingConfig, CosineLR, class LCNTraining
    LCNTraining.cpp
```

CMake target LCN, a static library with no further links. Predictor,
Decoder, and Head link it. Nothing in LCN includes Encoder.

## Interface

```cpp
struct LCNConfig
{
    size_t   dim;           // cube dimension; N = 2ᵈⁱᵐ; [4, 24]
    uint64_t seed;          // weight draw
    size_t   z_max;         // depth; 0 → dim, else >= 2
    size_t   gather_span;   // lookback window width; [2, 6]
    bool     tanh_last;     // false: last depth writes the raw sum
};

class LCN
{
public:
    static std::unique_ptr<LCN> Create(const LCNConfig& cfg);

    void Forward(std::span<const float> input_field);  // length N
    const std::vector<float>& Output() const;          // until the next Forward

    const std::vector<float>& Weights() const;         // depth, axis, tap, vertex
    void LoadWeights(std::span<const float> w);        // exact length; invalidates gradient

    size_t N() const;
    size_t Dim() const;
    LCNConfig Config() const;                          // z_max already resolved
};

struct LCNTrainingConfig
{
    float lr;                 // Adam step size; cosine peak; finite, > 0
    float lr_min_frac;        // floor = lr * lr_min_frac; in [0, 1]; 1 = constant
    int   lr_decay_epochs;    // cosine horizon; 0 = the num_epochs given to SetEpoch
    bool  restore_best;       // snapshot weights on a new low metric
    float beta1;              // Adam first-moment decay; in [0, 1)
    float beta2;              // Adam second-moment decay; in [0, 1)
    float eps;                // Adam denominator floor; finite, > 0
};

float CosineLR(float lr_max, float lr_min, int epoch, int num_epochs);

class LCNTraining
{
public:
    explicit LCNTraining(LCN& core, const LCNTrainingConfig& cfg);

    float Loss(std::span<const float> target);  // 0.5 × SSE; target may be a prefix
    void  Backward();                           // consume that Loss; sum into the gradient
    void  ZeroGrad();
    void  Adam();                               // one step; invalidates the loss seed
    void  AddGrad(std::span<const float> g);    // reduce a replica shard onto this trainer

    void  SetEpoch(int epoch, int num_epochs = 0);
    void  Observe(float metric, int epoch);     // restore_best: lower wins, strict
    void  RestoreBest();
    void  Reset();                              // forget Adam, lr, best; keep weights

    const std::vector<float>& Grad() const;     // same layout as Weights
    float Lr() const;
    const LCNTrainingConfig& Config() const;
};
```

Validation at Create: dim in [4, 24], resolved z_max at least 2 and
not so large the weight count overflows, gather_span in [2, 6].
LCNTraining rejects a non-finite lr or eps, lr_min_frac outside
[0, 1], betas outside [0, 1). The finiteness test is on the bits,
since std::isfinite is unreliable under fast-math. Forward throws if
the field is not length N. Loss throws if the target is empty or
longer than N.

## Contract summary

- Forward is one field in, one field out. The Output pointer is
  invalid after the next Forward.
- Each Forward starts the lookback stack over. Forwards do not chain.
- The caller's field is not modified.
- Loss then Backward must follow the Forward they belong to. Adam,
  LoadWeights, and RestoreBest all invalidate the loss seed.
- The gradient is a sum over the batch, not a mean.
- Config reports the resolved z_max, not a stored 0.
- One LCN is not thread-safe. Replicas are separate LCN plus
  LCNTraining pairs that share weights through LoadWeights and reduce
  with AddGrad.
- Predictor, Decoder, and Head each own an LCN. Head is the scalar
  readout for a planner.

The Predictor is [predictor.md](predictor.md). The Decoder is
[decoder.md](decoder.md). The WorldModel owns a Predictor, not an LCN
directly: [world_model.md](world_model.md).
