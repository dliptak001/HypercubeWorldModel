# Terrain walker

**Status: Release paired runs of tests/TerrainWalker/TerrainWalkerTest.cpp, dim = 8, T = 16, k = 5, 6, 7. Sine elevation. E(a) from a k-cube action encoder. Swap check: P reads a. The tables were produced before the LCN weight layout changed (Sep 10, 2026); the same seeds now draw a different net, so a re-run gives different numbers with the same picture.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Terrain walker | This world: a map, a 16×16 crop, and wander-then-acquire walks. The WorldModel predicts the next crop given the executed step. |
| WorldModel | Two frozen Encoders, one for the view and one for the action, plus a trained Predictor. See [world_model.md](world_model.md). |
| Encoder | Frozen hypercube reservoir. The view encoder: one field of N floats in, one episode, N floats out; the k-face is kept. |
| action encoder | A second Encoder with the same knobs and seeds on a k-cube. 2ᵏ floats in, 2ᵏ floats out, all kept. |
| Predictor | LCN on a cube of dimension k+1 (N = 2ᵏ⁺¹, twice the k-face). Input is E(x) cat E(a). The predicted next k-face is the first subcube of the output. |
| Decoder | Reconstruction meter. Not in the training loop. |
| dim | Encoder cube dimension. These runs: 8. |
| N | 2ᵈⁱᵐ = 256. Length of one observation. |
| k | Dimension of the kept face. Strictly less than dim. These runs: 5, 6, 7. |
| sub | Kept vertices, sub = 2ᵏ. |
| T | Encoder passes per episode, view and action alike. These runs: 16. |
| leak | Encoder leak rate. These runs: 0.25. |
| in_scale | Encoder input scaling. These runs: 0.8. |
| z_max | Predictor LCN depth. These runs: 3k. |
| span | Predictor LCN lookback. These runs: 5. |
| tanh_last | Last LCN depth applies tanh. These runs: on. |
| lr | Adam step size. These runs: 0.03. |
| epochs | Training passes over the mix. These runs: 1600. |
| view | The observation: a 16×16 crop, row-major onto the cube. Vertex i = row×16 + col. |
| map | A larger elevation grid the view is cut from. These knobs: 64×64. |
| N_P | Predictor vertices, 2ᵏ⁺¹ = 2 × 2ᵏ. |
| cell | One map square. Elevation from a few 2D sines, rescaled to [−0.5, 0.5]. |
| snake | The camera. A single point: the center cell of the view. No sprite. |
| pose | Snake / camera cell (row, col). Crop rows [r−8, r+7], cols [c−8, c+7]. |
| heading | Direction of travel: north, east, south, or west. |
| barrier | A map cell painted −1. The snake cannot occupy it (center overlay). Seeing a barrier off-center is not a hit. |
| map edge | The rim of the map. The 16×16 view must stay fully on-map. Touching that rim with any view edge is an encounter. |
| legal step | A cardinal move whose next crop stays on-map and whose next center is not a barrier. |
| goal | One free cell, not too close to the map rim, so the view can be centered on it. Painted +1 when it falls inside the crop. |
| frame x | The 16×16 view as N floats. Elevation in [−0.5, 0.5]; goal +1 if in view; barrier −1. The view never hangs off the map. |
| action a | The step actually taken this tick: north, east, south, or west. |
| ActionField | The action picture. A strip of 2ᵏ cells, 2ᵏ/16 rows by 16 columns, row-major: a half-plane of ones toward the heading, zeros elsewhere. |
| E(a) | EncodeAction of that strip: the whole action-cube output. Extra bit-face of P. |
| za | The buffer holding E(a). Length 2ᵏ. Four of them, one per heading, encoded once. |
| action_scale | Multiplier on E(a) at Pack. The raw action code runs hotter than the view k-face; this knob sets their ratio as P sees them. These runs: 0.33, which puts the two halves of the P input at the same mean level. |
| Heading | North, east, south, west. |
| TerrainMap | Elevation grid, barriers, goal, crop. |
| Walker | Camera pose and wander-then-acquire. Generates walks; not a learned policy. |
| Walk | Frames, executed headings, encounter flags, and heading-change flags. |
| encounter | Intended heading was illegal: 90°/180° recover. Printed as turn. |
| straight | Not an encounter. Includes keep-heading and legal acquire redirects. |
| keep | Executed cardinal equals the heading before the step. |
| change | Executed cardinal differs from the heading before the step. |
| L | Walk length in steps. These runs: 16. |
| pair | Consecutive crops on one walk, with the executed a. Never across maps. |
| mix | Train maps. These runs: 640 maps, 10240 pairs. |
| val | Unseen maps for restore-best only. These runs: 128 maps, 2048 pairs. |
| test | Held-out maps, not used to pick weights. These runs: 128 maps, 2048 pairs. |
| mse/ident | Pool MSE divided by identity. |
| mean |s| | Mean absolute value on a k-face. Live-code check. |
| wander | Until the goal is in the view: keep heading if that step is legal; otherwise turn 90° and step. |
| acquire | Once the goal is in the view: step to center the crop on the goal, still turning 90° at barriers and edges. |
| identity | Using E(xₜ) as the guess for E(xₜ₊₁). Copy-last. |
| next-power | Mean square of E(xₜ₊₁). Guessing nothing. |
| mse/power | Predictor residual vs next-power. The score. |
| first subcube | Vertices whose new address bit is 0: a contiguous prefix of length 2ᵏ. E(x) in, predicted next E(x) out. |
| extra bit-face | Vertices whose new address bit is 1. Holds E(a). |
| swap | The a-usage check. The same E(x) through P four times, once with each za. Evaluation only; P is not trained on it. |
| mse_true | Swap error with the executed heading. Equals the pool mse. |
| mse_wrong | Swap error averaged over the three headings that were not executed. |
| wrong/true | mse_wrong divided by mse_true. 1 means P ignores a. |
| spread | Mean square distance between P's guess with the true heading and its guess with a wrong one. 0 means P ignores a. |
| argmin==a | Fraction of pairs where the heading with the lowest swap error is the executed one. Chance is 25 %. |

Look at test mse/power: you want it as low as you can get, with val mse/power about the same number. Look at the swap line to know whether P reads a.

## The world

The Terrain walker is a 64×64 map of elevations, barriers, and one goal. The snake is a **point**: the camera center. The 16×16 view is always centered on that point. It wanders the bounded region, reacting to what it hits, until the goal appears in the frame. Then it walks the center onto the goal, still reacting to barriers and edges.

It does not sit still. If the next cell in the direction of travel is illegal, it **turns 90°** (left or right) and steps that way. Same rule for a map edge (any view edge would leave the map) and for a barrier (next center would sit on −1).

v1 still does not train a learned policy. This wander-then-acquire rule is how we **generate** walks. The world model only has to predict the next view given the step that was actually taken.

## Encounters

Two different hits, one response:

- **Map edge.** If any edge of the next 16×16 would leave the map, that heading is illegal. The view never hangs off. Off-map is not a pixel.
- **Barrier.** If the next **center** cell is −1, that heading is illegal. A −1 anywhere else in the picture is only paint.

When the current heading is illegal, pick a heading 90° from travel (left or right, uniform among those that are legal) and take that step. If both 90° turns are illegal, try 180°. If nothing is legal (boxed in), skip the step — should be rare if the map is not a cage.

a is the cardinal that was **executed**, not the heading you wished you had.

## Acquire

The goal is placed in the interior so the camera can sit on it: for a 64×64 map and this crop, pose must be able to reach that cell, so the goal lives in [8, 56]×[8, 56], not a barrier.

Until the goal cell is inside the current crop, **wander** (keep heading, turn 90° when blocked).

Once the goal is in the frame, **acquire**: prefer the cardinal that reduces Manhattan distance from pose to goal, if that step is legal; if not, the same 90° turn rule. Repeat until pose equals the goal, or the walk length runs out.

## One step

1. Draw a map. Elevations: a few 2D sines, rescaled to [−0.5, 0.5]. Barriers at random; start and goal free; goal interior. Start pose on-map for the crop ( [8, 56]² ) and not a barrier. Random initial heading.
2. Frame x is the crop. Goal in crop → +1. Barriers in crop → −1. Else elevation.
3. Encode x. EncodeAction the action strip. Pack is E(x) cat E(a). The cardinal is not in the view pixels.
4. Choose the executed step from wander or acquire as above. Move pose. Next frame is the new crop.

## The action picture and its code

The heading has to reach P as a code the same length as E(x), which is 2ᵏ. It gets there the same way the view does: paint a picture, run a frozen reservoir on it, keep the output.

**The picture.** A strip of 2ᵏ cells laid out as 2ᵏ/16 rows by 16 columns, row-major, so the strip is a shorter version of the view's 16-column layout. At k = 5 that is 2 rows, at k = 6 it is 4, at k = 7 it is 8. North paints the top half of the rows with ones, south the bottom half, west the left half of the columns, east the right half. Everything else is zero. Half the strip carries signal for every heading, so each of the four pictures is fat, and they are four distinct pictures.

**The encoder.** The WorldModel builds a second Encoder from the view encoder's config with dim replaced by k. Same leak, same input scaling, same passes, same seeds. Its cube has 2ᵏ vertices, one per cell of the strip, so the strip is its whole input and its whole output is already 2ᵏ long. Nothing is cut. That output is E(a).

**Four codes, made once.** Each heading's picture is encoded once at startup into its za buffer. Every pair on every walk reuses the code for the heading that was executed.

**The scale.** The whole output of a small cube comes out louder than a face cut from a big one. action_scale is a multiplier that Pack applies to E(a) as it lays it on the Predictor cube, so P sees the two channels at a chosen ratio. The four za buffers are never changed; the swap check reads the same raw codes whatever the scale.

## Putting a on a bigger cube

P **grows**. Encoder k-face stays length 2ᵏ. Predictor dim is **k+1**, so N_P = 2ᵏ⁺¹ = 2 × 2ᵏ.

The extra address bit splits the cube in half:

- **First subcube** (new bit = 0): E(x), untouched. Length 2ᵏ.
- **Extra bit-face** (new bit = 1): action_scale × E(a). Length 2ᵏ. Vertex i of the view sits one hop from vertex i of the action channel, so the LCN's first depth already sees both.

E(x) is not scaled or added into. The two channels are the same length but come from different cubes: E(x) is the k-face of the view encoder's dim-cube, E(a) is the whole output of the action encoder's k-cube.

LCN still maps the full doubled cube to itself. **Loss and Predict use only the first subcube** of that output — the next k-face, length 2ᵏ. LCNTraining already allows a target shorter than N: the extra bit-face is unconstrained.

The Decoder never sees a. The action side has no cut and no Decoder; the action cube is already the k-face size.

If P ignores E(a), north and east from the same view produce the same guess. They must not. The swap check below measures exactly that.

## Maps and data

Many maps, like many songs. On each map, a walk of L steps under wander-then-acquire. Pairs are (E(xₜ), E(aₜ)) → E(xₜ₊₁) inside one map. Never across maps.

No requirement that acquire finishes before L. Short walks that never see the goal are still valid wander data.

## What you train and score

Frozen Encoder, same knobs as the two-sine world (dim = 8, T = 2×dim, leak = 0.25, in_scale = 0.8). Predictor learns P(E(x) cat E(a)). Decoder is not in that loop. Span 5, tanh_last on, lr = 0.03, 1600 epochs, action_scale 0.33. Only k changes across the rows; Predictor depth is z_max = 3k.

Same walk mix at every k: 640 / 128 / 128 maps, L = 16, 10240 train pairs, 1396 turns, 100 of 640 walks saw the goal. Each slice (straight, turn, keep, change) uses its own identity and next-power.

Straight vs turn is **encounter vs not**. Acquire can legally change cardinal and still count as straight, so that split is the wall bar. Keep vs change is whether the executed heading flipped. Neither split says whether P reads a: P never sees the heading before the step, so a P that reads a and a P that ignores a both leave the same residual on keep as on change. The earlier reading of this table, that matching keep and change numbers meant P was ignoring a, was wrong for that reason.

The swap check is the a bar. Take one test pair, run the same E(x) through P four times with each za, and score each guess against the real next code. If P ignores a, all four guesses are the same, wrong/true is 1, and the best-scoring heading is the real one a quarter of the time. If P reads a, the true heading scores lowest and wrong/true climbs.

| k | sub | test mse/power | val mse/power | test mse/ident | keep / change | straight / turn | wrong/true | argmin==a |
|---|-----|---------------:|--------------:|---------------:|--------------:|----------------:|-----------:|----------:|
| 5 | 32 | 0.108 | 0.116 | 0.39 | 0.108 / 0.108 | 0.109 / 0.102 | 3.21 | 93.55 % |
| 6 | 64 | 0.077 | 0.083 | 0.28 | 0.076 / 0.080 | 0.077 / 0.075 | 4.72 | 98.88 % |
| 7 | 128 | 0.072 | 0.077 | 0.26 | 0.072 / 0.075 | 0.073 / 0.070 | 4.70 | 99.27 % |

Test swap detail:

| k | mse_true | mse_wrong | spread | identity | next-power |
|---|---------:|----------:|-------:|---------:|-----------:|
| 5 | 0.00256 | 0.00822 | 0.00568 | 0.00659 | 0.02378 |
| 6 | 0.00158 | 0.00747 | 0.00588 | 0.00567 | 0.02068 |
| 7 | 0.00148 | 0.00694 | 0.00534 | 0.00559 | 0.02043 |

The next-crop job is real. Held residual falls as k grows, val tracks test, and the codes stay live (mean |s| about 0.09 on a train crop). Copy-last is a strong guess, because consecutive crops overlap almost completely, and P still beats it by 2.6× to 3.8×.

The heading job is real too. Lying to P about the heading costs 3.2× to 4.7× the error, and the wrong-heading error sits above copy-last: a guess made with the wrong heading is worse than not moving at all. Given all four headings, P picks the executed one 93.6 % to 99.3 % of the time, against 25 % by chance. Growing the face makes it sharper. Keep and change still match, exactly as they must whether or not P reads a.

The four E(a) codes come out at similar strength, mean absolute 0.24 to 0.33, because the action cube is the strip's own size and every heading paints half of it. Nothing is cut on the action side, so no heading lands on a dead part of the code.

Identity in code space is about a quarter of next-power. Part of that is the smooth terrain, and part is a fixed vector that every crop shares, about a third of the code's power, from the same s0 and a spectral radius near 1. That shared part inflates next-power, not identity.

Semantic junk-vs-structure, later. Not v1.

## Product face

WorldModel Pack is two channels:

- Encode(field, z) for the view: N in, k-face out. EncodeAction(strip, za) for the action: 2ᵏ in, 2ᵏ out, its own k-cube encoder.
- Predict(z, za) and Accumulate(z, za, z_next): both codes length 2ᵏ. Internally cat onto a (k+1)-cube. Return and loss are the first subcube (2ᵏ).

Knobs at the head of tests/TerrainWalker/TerrainWalkerTest.cpp: map size, view 16, barrier fraction, walk length, map counts, action scale, Encoder/Predictor knobs. Predictor dim is k+1 and the action cube is k; neither is a separate knob. The half-plane paint is ActionField.

## Settled

- Camera = snake = one point, the crop center. No sprite.
- Elevation [−0.5, 0.5], goal +1, barrier −1.
- Crop [r−8, r+7]×[c−8, c+7], always fully on-map.
- Map edge: view edge would leave the map → that heading is illegal.
- Barrier: next center would be −1 → that heading is illegal.
- Illegal heading → turn 90° and still step. No sit-still.
- Wander until the goal is in the view; then stepwise center on the goal, still turning at encounters.
- Goal placed so centering is possible (interior of the legal pose rectangle).
- Map 64×64, walk length 16, barrier fraction ~0.12.
- When both 90° turns are legal, pick left or right uniformly.
- Predictor dim = k+1 (N_P = 2 × 2ᵏ). No add-into E(x). One scale on E(a) at Pack.
- Input: E(x) cat E(a). Action field is a 2ᵏ strip, half-plane of ones toward the heading, through its own k-cube encoder.
- Output and loss: the first subcube of P only, length 2ᵏ. The extra bit-face is unconstrained.
- a-usage is read from the swap check, never from the keep / change or straight / turn slices.
- The test fails if the wrong heading does not score worse than the true one on test.

## Files

```
tests/TerrainWalker/
    Heading.h              cardinals, turns, step delta
    Map.h / Map.cpp        TerrainMap: draw, crop, pose range
    Walker.h / Walker.cpp  Walker, Walk, RecordWalk
    ActionField.h / .cpp   half-plane paint on a rows × cols strip
    TerrainWalkerTest.cpp  knobs, mix, train, held mse/power, slices, swap check
```

CMake target TerrainWalkerTest, linking WorldModel and ThreadPool.

No learned policy in v1. No OpenMP. No CLI.
