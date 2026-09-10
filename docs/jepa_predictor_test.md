# JEPA predictor test

**Status: tests/JepaPredictorTest.cpp implemented. No Release numbers yet.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Encoder | Frozen hypercube reservoir. One window in, one episode, N floats out. |
| Predictor | LCN on the k-face: E(xₜ) in, predicted E(xₜ₊₁) out. |
| Decoder | Reconstruction meter. Not used. |
| dim | Encoder cube dimension. This test: 8. |
| N | Window length and encoder vertices, N = 2ᵈⁱᵐ = 256. |
| k | Dimension of the kept face. Strictly less than dim. This test: 7. |
| k-face | The low 2ᵏ vertices of the encoder's newest slice. |
| E(x) | That k-face for window x. |
| stream | One long two-sine. Frequencies, phases, and amplitudes are drawn once. |
| window | N consecutive samples of the stream, laid on the cube (vertex i = sample i). |
| hop | Samples between window starts. This test: hop = N, so windows do not overlap. |
| identity | MSE of using E(xₜ) itself as the guess for E(xₜ₊₁). Copy-last. |
| next-power | Mean square of E(xₜ₊₁). MSE of predicting nothing. |
| test MSE | Mean square of P(E(xₜ)) vs E(xₜ₊₁) on the held-out windows. |
| test / identity | test MSE divided by identity. The prediction score. |
| ok | Process exit: train MSE fell, and test MSE beat next-power. |

The test cuts one two-sine stream into consecutive windows, encodes each
from a fresh episode, and trains the Predictor on adjacent k-faces.
Train, val, and test are later stretches of that same stream. The
Decoder is not in the loop.

`ok` is not the readout. Beating next-power only says the guess is
better than silence. A k-face with a standing mean can pass that
without tracking the next window. The printed line to read is
`test / identity`, then identity against next-power.

| test / identity | Meaning |
|-----------------|--------|
| clearly below 1 | The Predictor learned a phase advance. Copy-last loses. |
| about 1 | The Predictor matched persistence. It did not beat copy-last. |
| above 1, and the process still printed ok | Worse than copy-last. It beat zero anyway. |

Identity vs next-power is the sanity check on the hop. With hop = N
the next window is a large phase jump (cycles in [0.7, 7.3] turns
per window), so identity and next-power should sit in the same
ballpark: consecutive k-faces are not near copies of each other. If
identity is tiny against next-power, the faces barely move and the
task has collapsed; beating identity would then be the wrong story.

This is one oscillator, split in time. A low test / identity means
P can track this stream's k-faces. It does not mean a new pair of
sines would follow. Training on many draws and scoring a held-out
draw is [world_model_test.md](world_model_test.md).
