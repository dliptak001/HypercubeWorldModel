# Compression test

**Status: Release runs of tests/CompressionTest.cpp, dim = 8. The tables were produced before the LCN weight layout changed (Sep 10, 2026); the same seeds now draw a different net, so a re-run gives different numbers with the same picture.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Encoder | Frozen hypercube reservoir. One field in, one episode, N floats out. |
| Decoder | Trained LCN. k-face in, reconstructed field out. |
| dim | Cube dimension. These runs: 8. |
| N | Vertices and field length, N = 2ᵈⁱᵐ = 256. |
| k | Dimension of the kept face. Strictly less than dim. |
| sub | Kept vertices, sub = 2ᵏ. |
| field | One window of N samples on the cube, vertex i = sample i. |
| term | One sine in that window. Each field is the sum of two terms. |
| cycles | Oscillations of a term across the N samples. Drawn in [0.7, 7.3], not integer, so a window is not one period. |
| phase | Starting angle of a term, uniform over a full turn. |
| amplitude | Scale of a term, drawn in [0.3, 1]. |
| leak | Encoder leak rate. These runs: 0.25. |
| in_scale | Encoder input scaling. These runs: 0.8. |
| k-face | The low sub vertices of the encoder's newest slice. |
| input scale | Decoder constant, 1 / max \|k-face\| on the training set. |
| MSE | Mean squared error of Decode vs the original field, averaged over vertices and windows. |
| field power | Mean square of the field. These runs: 0.47638. |
| compression | N / sub. |

Each field is two sines added on the cube, time order along vertex index. For each term, cycles, phase, and amplitude are drawn independently from the ranges above. Splits share one data seed, so every k row sees the same windows.

CompressionTest encodes each field, keeps the k-face, trains the Decoder to emit the original field, and scores train / val / test MSE. Encoder weights and s₀ are frozen. Same encoder and same fields in every row; only k changes. Train 1024, val 512, test 512, 400 epochs.

| k | sub | compression | train MSE | val | test | test / 2× |
|---|-----|-------------|----------:|----:|-----:|----------:|
| 7 | 128 | 2× | 0.00305 | 0.00380 | 0.00434 | 1 |
| 6 | 64 | 4× | 0.00769 | 0.00887 | 0.00993 | 2.29 |
| 5 | 32 | 8× | 0.01706 | 0.02218 | 0.02395 | 5.52 |
| 4 | 16 | 16× | 0.03603 | 0.05098 | 0.05410 | 12.5 |

Test MSE × sub is about 0.56, 0.64, 0.77, 0.87. Reconstruction error still grows as k shrinks, a bit steeper than 1 / 2ᵏ at the small faces. Residual energy vs field power: 0.9 %, 2.1 %, 5.0 %, 11.4 %. At k = 4 the train–val gap opens; val still tracks test.

The encoder cube stays live (mean \|z\| = 0.2298). Decoder input scale stays ~1.01. The extra compression is not a gain trick.

These rows are T = 40. The T = 16 encoder used with the world model, and the semantic noise/content meter at that T, are [world_model_test.md](world_model_test.md) and [jepa_encoder_test.md](jepa_encoder_test.md).
