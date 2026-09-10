# JEPA encoder test

**Status: Release runs of tests/JepaEncoderTest.cpp, dim = 8, T = 16. The tables were produced before the LCN weight layout changed (Sep 10, 2026); the same seeds now draw a different net, so a re-run gives different numbers with the same picture.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Encoder | Frozen hypercube reservoir. One field in, one episode, N floats out. |
| dim | Encoder cube dimension. These runs: 8. |
| N | Field length and encoder vertices, N = 2ᵈⁱᵐ = 256. |
| k | Dimension of the kept face. Strictly less than dim. These runs: 7, 6, 5. |
| sub | Kept vertices, sub = 2ᵏ. |
| k-face | The low sub vertices of the encoder's newest slice. The candidate latent. |
| T | Encoder passes per episode. These runs: 16. |
| field | Two-sine window on the cube, same generator as [compression_test.md](compression_test.md). |
| content | A different two-sine draw. Structure change. |
| noise | i.i.d. Gaussian N(0, σ) added to the same field. These runs: σ = 0.2. |
| RMSE | Root mean square difference, per coordinate. |
| noise/content | Mean RMSE(A, A+noise) / mean RMSE(A, B) over 1024 pairs. |
| rel | (noise/content) on the k-face divided by (noise/content) on the field. |
| leak | Encoder leak rate. These runs: 0.25. |
| in_scale | Encoder input scaling. These runs: 0.8. |
| dimensional compression | Fewer numbers: keep sub vertices out of N. Ratio N / sub. True by the cut. |
| semantic compression | The kept numbers are the ones that matter: structure stays, junk is dropped. |

This test is a semantic meter. It is not a dimensional one.

Dimensional compression is the k-cut: 256 values become 128, 64, or 32. That is proven as soon as you throw away the rest of the cube. Reconstruction and next-step, in [world_model_test.md](world_model_test.md), ask whether that smaller list still carries the two-sine and still has a next face. They do not ask whether the list is a filter.

Semantic compression is the filter question. Take a field A, the same field plus white noise, and a different two-sine B. Measure how much the noise moves A, and how much a real song change moves A, first on the samples and then on the k-face. rel is that noise/content ratio on the k-face, divided by the same ratio on the field.

rel is the score. The Decoder is not used.

- rel ~ 1: the k-face copies the field. Dimensional cut of the waveform, not a filter.
- rel > 1: the k-face boosts junk vs structure. Bad latent.
- rel clearly below 1, with content RMSE still large: noise is damped more than a real signal change. Semantic. JEPA-shaped.
- rel ~ 0 and content RMSE collapsed: the code went dead, including the sines.

A viable encoder on this meter is the third case, read next to reconstruction: the sines must still be in the k-face, and white noise must move that face less, relative to a new sine, than it moves the samples.

Same encoder as the T = 16 world-model / compression pair. Only k changes. Field noise/content is 0.2113 on every row (same 1024 pairs, same σ).

| k | sub | N / sub | k-face noise | k-face content | k-face noise/content | rel |
|---|-----|--------:|-------------:|---------------:|---------------------:|----:|
| 7 | 128 | 2× | 0.03575 | 0.426 | 0.0839 | 0.397 |
| 6 | 64 | 4× | 0.03611 | 0.424 | 0.0851 | 0.403 |
| 5 | 32 | 8× | 0.03861 | 0.447 | 0.0864 | 0.409 |

rel is ~0.40 at every cut. Content RMSE stays ~0.42–0.45: different sines still separate. The filter is the leaky orbit, not how many vertices you keep. Dimensional compression changes across the table; semantic compression does not. You are not getting more semantic compression by keeping a smaller face.

A T = 40 sweep at 256 pairs printed rel = 0.39 at k = 7..4, same pattern. Leak = 1 at that drive was rel ≈ 0.97 (isometry: a smaller copy of the field). Leak 0.25 is the switch that makes the code a filter.

**Where we stand.** At leak 0.25, in_scale 0.8, dim 8, T = 16, the frozen k-face damps this junk (white noise) relative to this content (two sines). That is the semantic claim this test can prove, and it proves it. Reconstruction at the same knobs still recovers the sines, so the sines were not the thing that got damped. Next-window prediction in k-face space is [world_model_test.md](world_model_test.md). The Decoder is not in that loop.
