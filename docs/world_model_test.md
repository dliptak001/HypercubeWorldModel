# World model test

**Status: Release paired runs of tests/WorldModelTest.cpp and tests/CompressionTest.cpp, dim = 8, T = 16. Table is the k-cube Predictor (no a). Not re-run since WorldModel grew a. The tables were produced before the LCN weight layout changed (Sep 10, 2026); the same seeds now draw a different net, so a re-run gives different numbers with the same picture.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| WorldModel | Frozen Encoder plus trained Predictor. Does the k-cut. See [world_model.md](world_model.md). |
| Encoder | Frozen hypercube reservoir. One window in, one episode, N floats out. |
| Predictor | LCN. This document's table is the k-cube map P(E(xₜ)) → E(xₜ₊₁). WorldModel now uses dim = k+1 and Pack is E(x) cat E(a); this test runs a constant dummy field of length 2ᵏ through the action encoder (no motor). |
| a | Action field, length N. Constant fill here; how to paint a cardinal is later. |
| E(a) | EncodeAction of that field: the whole output of a k-cube encoder. Extra bit-face of P. |
| Decoder | Reconstruction meter. k-face in, full field out. Not in the WorldModel loop. |
| dim | Encoder cube dimension. These runs: 8. |
| N | Window length and encoder vertices, N = 2ᵈⁱᵐ = 256. |
| k | Dimension of the kept face. Strictly less than dim. These runs: 7, 6, 5. |
| sub | Kept vertices, sub = 2ᵏ. |
| T | Encoder passes per episode. These runs: 16. |
| leak | Encoder leak rate. These runs: 0.25. |
| in_scale | Encoder input scaling. These runs: 0.8. |
| song | One two-sine draw: its own cycles, phases, amplitudes. |
| window | N consecutive samples of a song, laid on the cube. |
| hop | Samples between window starts. These runs: hop = N. |
| pair | Two consecutive windows of the same song. Never across songs. |
| mix | Train songs. These runs: 640 songs, 8 windows, 4480 pairs. |
| val | Unseen songs for restore-best only. These runs: 128 songs. |
| test | Held-out songs, not used to pick weights. These runs: 128 songs. |
| identity | MSE of using E(xₜ) as the guess for E(xₜ₊₁). Copy-last. |
| next-power | Mean square of E(xₜ₊₁). MSE of predicting nothing. |
| mse/ident | WorldModel pool MSE divided by identity. |
| mse/power | WorldModel pool MSE divided by next-power. Residual vs k-face energy. |
| field power | Mean square of the sine field. These CompressionTest runs: 0.47638. |
| dimensional compression | Fewer numbers: keep sub = 2ᵏ vertices out of N. Ratio N / sub. Proven by the cut. |
| semantic compression | The kept numbers are the ones that matter: structure stays, junk is dropped. At these encoder knobs: rel ≈ 0.40 on [jepa_encoder_test.md](jepa_encoder_test.md). |
| rel | (noise/content) on the k-face divided by the same ratio on the field. Semantic score. These runs: ~0.40 at every k. |

Look at test mse/power: you want it as low as you can get, with val mse/power about the same number.

The k-cut is dimensional compression: 256 numbers become 128, 64, or 32. That is true before any training. Semantic compression is the filter: the two-sine stays, white noise is damped more than on the raw samples. This document does not run that meter. The paired JepaEncoderTest at the same encoder does: rel ≈ 0.40 at k = 7, 6, and 5, and content RMSE stays large. Reconstruction and next-step here say the smaller face still carries the song and still has a next k-face. Together that is dimensional cut plus a live, filtered code.

The Encoder is frozen. WorldModelTest trains a Predictor on many two-sine songs and scores a held-out mix. CompressionTest, at the same encoder, trains a Decoder to put the field back from the k-face. Decoder is not in the WorldModel loop. Only k changes across the rows; Predictor depth is z_max = 3k. Span 5, tanh_last on, lr = 0.03. WorldModel 800 epochs, CompressionTest 1024 / 512 / 512 fields, 400 epochs.

| k | sub | N / sub | test mse/power | val mse/power | test mse/ident | recon test MSE | recon / power | rel |
|---|-----|--------:|---------------:|--------------:|---------------:|---------------:|--------------:|----:|
| 7 | 128 | 2× | 0.137 | 0.136 | 0.077 | 0.02141 | 4.5 % | 0.397 |
| 6 | 64 | 4× | 0.174 | 0.170 | 0.097 | 0.02453 | 5.1 % | 0.403 |
| 5 | 32 | 8× | 0.366 | 0.361 | 0.201 | 0.03651 | 7.7 % | 0.409 |

At k = 7, held residual is ~14 % of next-face energy and val is the same number. Copy-last loses by about 13×. The Decoder puts the field back to 4.5 % of field power. Dimensionally, half the cube is gone. What remains still holds the sines and still predicts the next face on new songs.

k = 6 is the same story, a little looser: held mse/power 0.174, val 0.170, reconstruction 5.1 %. Four times fewer latent vertices, almost the same next-step and almost the same reconstruction.

k = 5 still beats silence and copy-last (mse/power 0.37, mse/ident 0.20), and val still matches test. Reconstruction is 7.7 % of field power. The next-step residual has grown more than the reconstruction residual: the cut is starting to hurt prediction first.

WorldModel detail:

| k | train mse | train mse/power | val mse | test mse | identity (test) | next-power (test) |
|---|----------:|----------------:|--------:|---------:|----------------:|------------------:|
| 7 | 0.12275 → 0.00371 | 0.037 | 0.01273 | 0.01343 | 0.174 | 0.098 |
| 6 | 0.12748 → 0.00890 | 0.088 | 0.01631 | 0.01728 | 0.178 | 0.099 |
| 5 | 0.13725 → 0.02511 | 0.217 | 0.03909 | 0.04058 | 0.201 | 0.111 |

Train is tighter than held on every row. That gap is mix-fit, not a collapse: val and test stay together. Identity and next-power stay in the same ballpark (hop = N is still a real jump). k-faces stay live (mean |s| on a train window ~0.20–0.22; full-cube mean |z| = 0.213). Decoder input scale stays ~1.04, so the smaller face is not a gain trick.