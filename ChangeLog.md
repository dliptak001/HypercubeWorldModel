# HypercubeWorldModel - Change Log

## Unreleased
- Python SDK: restore_best defaults on for WorldModel and Decoder; WorldModel.passes and config()["passes"] are as given (0 stays 0); Head.predict (and Head(z)) is the scalar map; Head.plan_cost(zs) takes the rollout array; VectorModel.fit and the training cycle forward to the WorldModel. VectorModel(dim=, k=, ...) builds the WorldModel; Head.fit uses batch_size (keyword-only epochs); action_low/high must be paired; set_obs_norm / set_action_bounds after construct. Encode of a short vector names paint_stripes / VectorModel.encode.
- Host API: Predict and Decode write into a caller buffer, same as Encode. Config() keeps encoder.passes as given so Create(Config()) round-trips. VectorModel::Create(cfg) builds the WorldModel; the training cycle is on VectorModel. Head::Fit drops the redundant count (y.size()); za is last so epochs are easy to pass. Head::Predict is the scalar map (was Apply). PlanCost and Cost infer batch and H+1 from the spans. restore_best defaults on for WorldModel and Decoder.
- C++ SDK thinning: RequireLastDim is no longer public (VectorModel still names both sizes in last-dim errors). WorldModel stream Save/Load are private (VectorModel embedding only). LCN and LCNTraining pointer overloads are private; span is the public form. Predict/Pack/Accumulate take za. Head::Fit takes code_size; PlanCost, VectorModel::Cost, and RolloutErrorFromCodes take path_len (H+1).
- 1.1.0: VectorModel, Normaliser, Head, MinDim/MinK, and MSE/linR2 health functions in C++ (third product library). VectorModel Save uses a new magic (`HVM1`); `HWM1` is unchanged. Python bindings wrap the C++ types. WorldModel without VectorModel is still the field/code API. SetHead rejects an unfitted Head. Head is an LCN on the code cube (readout is vertex 0); code_size a power of two, at least 16. HVM1 file version 2. Head AUC only when y is strictly 0/1 with both classes (ties 0.5). Action sensitivity samples in stored action bounds if set, else [-1, 1]. VectorModel.rollout last-dim must match act_dim once set. lin_r2 is an SGD linear map, no closed-form solve.
- Encoder: output_scale presentation gain on the returned cube (default 1, not drive). Delay line stays raw (RawCube). SuggestedOutputScale is target_rms / rms(raw); FitOutputScale sets it; Suggest does not mutate. WorldModel::LastCube is the scaled Encode return; LastRawCube is unscaled.
- WorldModel: drop action_scale. Pack is concat. Each encoder has its own output_scale. HWM1 file version 2 writes both scales; Load of v1 maps the old Pack multiplier onto the action encoder and leaves the view at 1. Python WorldModel(..., action_scale=) is gone; a 1.0.0 pickle that stored it maps the same way.
- Magnitude readouts: MeanAbs and Rms on a span; Encoder RawCube / ScaledCube; WorldModel LastRawCube / LastPacked.
- Planner example and SDK prose name the surface as a planner protocol, not after a suite. Types stay WorldModel / Decoder / paint_stripes.
- Drop the elevation-map walker study (test target, sources, and write-up). The suite sandbox is the action-conditioned host.
- Drop tests/ (CompressionTest, JepaEncoderTest, JepaPredictorTest, WorldModelTest) and their write-ups. Smoke is main.cpp; Python contract tests remain.
- GitHub Release v1.0.0 created from the existing tag with the ChangeLog entry as notes and the 25 wheels and the sdist from the tag's workflow run attached
- wheels.yml: the publish job creates the GitHub Release on every v* tag, wheels and sdist attached, notes auto-generated; a tag alone never created one

## v1.0.0 (Sep 10, 2026)
- Released: tag v1.0.0, wheels for Python 3.10 to 3.14 on Windows x64, Linux x86_64 and aarch64, and macOS x86_64 and arm64, plus the sdist, published to PyPI as hypercube-worldmodel 1.0.0 through the trusted publisher with digital attestations
- Repository made public with a fresh history; README badges for the wheels build, PyPI, and Python versions on both pages
- Python planner adapter matches the DMC protocol: rollout takes raw actions and encode_action's the block once; action_space.low / .high for CEM. State-based DMC is the documented path; paint_stripes is not a pixel encoder
- Python SDK: pip package hypercube-worldmodel, import hypercube_worldmodel, with WorldModel, Decoder, and paint_stripes; every batched call loops in C++ with the GIL released; fit with an optional validation set; save and load share the C++ file formats, pickle for convenience; docs/Python_SDK.md, python/README.md, two examples, a 34-test suite, and a cibuildwheel workflow. The version is one string in python/hypercube_worldmodel/_version.py, checked at compile time against WorldModel::kVersion
- WorldModel::RequestedPasses returns encoder.passes as given to Create, for hosts that serialize their own config
- Decoder gains Weights, LoadWeights, Grad, and AddGrad, matching WorldModel
- API: FieldSize and CodeSize on WorldModel and Decoder, k on both configs; Predictor knobs nested as WorldModelConfig.predictor including training; EncoderConfig.verbose dropped; WorldModel::kVersion; RealizedSpectralRadius without a Get prefix; Encoder::Create exceptions named
- LCN Forward and Backward run the vertex loop innermost over a weight layout of depth, axis, tap, vertex, so the weights of one tap are contiguous and the gradient scatter is no longer a strided read-modify-write: about a fifth off Forward and close to half off a training step from dim 6 to 12. Same function, checked against the old code with identical weights. The layout differs from HypercubeLCN, so weights do not move between the two without a transpose. The same seed now draws a different net; the results tables in docs predate this and re-run to different numbers
- ThreadPool constructor is exception-safe: if a worker thread cannot be started, the workers already running are stopped and joined before the exception propagates
- Encoder::Create rejects a non-finite spectral_radius, leak_rate, or input_scaling, and WorldModel::Create a non-finite action_scale, with a bit-level test that holds under fast-math; the main smoke test covers all four
- Encoder accepts dim up to 24, matching the LCN; every cube in the family now has the same ceiling
- C++ SDK: docs/CPP_SDK.md in the family format, with terms defined inline, the cycle, the API map, the replica recipe, encode costs, and a section on driving the model from a planner, including the DeepMind Control Suite latent world model convention
- WorldModel: Save writes encoder.passes as given to Create, not the resolved view T, so Load rebuilds the action encoder with the same T; the main smoke test round-trips a passes = 0 model and compares E(x), E(a), and Predict
- WorldModel: z_max defaults to 0, meaning k+1, instead of a fixed depth bound to the default k
- WorldModel: Rollout chains Predict over H action codes into caller storage, H + 1 view codes out
- WorldModel: Save and Load carry the config and the Predictor weights; the encoders are rebuilt from their seeds
- PaintStripes: a free function that lays a short state or action vector onto a field as contiguous stripes
- examples/quick_start.cpp: the SDK quick start as a CMake target, a point on a plane moved by a bounded velocity; test mse/power about one percent of identity, and a save/load round trip check
- CMake project VERSION 1.0.0
- README SDKs section points at the guide
- Decoder is a first-class SDK class beside WorldModel: its own section in docs/CPP_SDK.md, both targets in Build and link, README shows both; the Python SDK will carry the same two classes

## v0.1.1 (Sep 10, 2026)
- Terrain walker table re-run at k = 5, 6, 7 with action_scale 0.33 and 1600 epochs: test mse/power 0.108 / 0.077 / 0.072, wrong/true 3.2 / 4.7 / 4.7, argmin==a 93.6 / 98.9 / 99.3 %
- Docs: world_model.md gains a plain-language section on the action path; terrain_walker.md splits the action picture and its code from the Predictor cube layout
- WorldModel: action_scale knob, a multiplier on E(a) as Pack lays it on the extra bit-face; the stored action code is untouched. TerrainWalkerTest exposes it as kActionScale and prints both halves of the P input
- TerrainWalkerTest prints argmin==a to two decimals
- WorldModel: E(a) comes from a second Encoder on a k-cube, same knobs and seeds as the view encoder; EncodeAction takes a 2ᵏ field and writes its whole 2ᵏ output. k must be at least 5
- ActionField paints a rows × cols strip, so the terrain walker's action is a 2ᵏ half-plane instead of an N-float one; the four heading codes now have similar strength
- WorldModelTest dummy action field is 2ᵏ long through EncodeAction
- Docs: world_model.md, terrain_walker.md, world_model_test.md describe the action encoder; terrain walker table re-run at k = 5, 6, 7

## v0.1.0 (Sep 09, 2026)
- Add TerrainWalkerTest: 64×64 elevation map with barriers and a goal, a 16×16 crop as the view, wander-then-acquire walks, and a WorldModel trained on E(x) cat E(a) pairs across many maps
- Add the swap check to TerrainWalkerTest: the same E(x) through P with all four headings; fail if the wrong heading does not score worse than the true one, and print wrong/true, spread, and argmin==a
- Correct docs/terrain_walker.md: keep/change and straight/turn slices cannot show whether P reads a; the swap check shows it does
- Terrain elevation is three 2D sines (a uniform-per-cell trial was reverted); print stage scales for view, action field, and P input
- Add per-slice next-power and the keep vs change and straight vs turn splits to TerrainWalkerTest
- WorldModel: Predict(z, a) packs E(x) cat E(a) onto a (k+1)-cube; loss and return use the first subcube only
- Add WorldModel: frozen Encoder plus trained Predictor with the k-cut inside; Encode writes into caller storage
- Add WorldModelTest: train on a mix of two-sine songs and score a held-out mix, with a Predictor replica thread pool, wall times, and one score line per split
- Add Predictor: forward map from a k-face to the next k-face, with LCN training and JepaPredictorTest
- Rename CompressionTest2 to JepaEncoderTest with shared run banners; add encoder drive, depth, span, and epoch knobs to CompressionTest
- Docs: terrain walker design, world model and world model test, dimensional vs semantic compression, JEPA encoder and predictor tests, compression test k-sweep
