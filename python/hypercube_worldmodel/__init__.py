"""HypercubeWorldModel: a world model on a Boolean hypercube.

Two frozen hypercube reservoirs encode the view and the action, and a
locally connected net, the Predictor, learns to map the two codes to the
next view code. Only the Predictor trains. A separate Decoder turns any
code back into a field.

Quick start::

    import numpy as np
    import hypercube_worldmodel as hw

    wm = hw.WorldModel(dim=6, k=5, passes=12, leak_rate=0.25,
                       input_scaling=0.8,
                       z_max=15, gather_span=5, tanh_last=True,
                       lr=0.03, lr_min_frac=0.05, restore_best=True)
    z = wm.encode(hw.paint_stripes(obs, wm.N))          # (count, code_size)
    za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
    z_next = wm.encode(hw.paint_stripes(obs_next, wm.N))
    wm.fit(z, za, z_next, epochs=200, batch_size=16)
    hat = wm.predict(z[0], za[0])                        # (code_size,)
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np

from ._core import _Decoder, _WorldModel, cpp_version
from ._core import _Head, _Normaliser, _VectorModel
from ._core import paint_stripes as _paint_stripes
from ._core import mean_abs as _mean_abs
from ._core import rms as _rms
from ._core import min_dim as _min_dim
from ._core import min_k as _min_k
from ._core import no_change_mse as _no_change_mse
from ._core import one_step_ratio as _one_step_ratio
from ._core import action_sensitivity_from_preds as _action_sensitivity_from_preds
from ._core import lin_r2 as _lin_r2
from ._core import rollout_error_from_codes as _rollout_error_from_codes
from ._version import __version__

__all__ = [
    "WorldModel", "Decoder", "Normaliser", "Head", "VectorModel", "ActionSpace",
    "paint_stripes", "mean_abs", "rms", "min_dim", "min_k",
    "no_change_mse", "one_step_ratio", "action_sensitivity", "lin_r2",
    "rollout_error", "__version__",
]

if cpp_version != __version__:  # pragma: no cover
    raise ImportError(
        f"hypercube_worldmodel {__version__} was built against C++ core "
        f"{cpp_version}; reinstall the package"
    )


def _f32(a):
    """C-contiguous float32 view or copy."""
    return np.ascontiguousarray(a, dtype=np.float32)


def _rows(a, width, what):
    """Validate a (width,) or (count, width) array; return (array2d, was_1d)."""
    a = _f32(a)
    if a.ndim == 1:
        if a.shape[0] != width:
            raise ValueError(f"{what} must have length {width}, got {a.shape[0]}")
        return a.reshape(1, width), True
    if a.ndim == 2:
        if a.shape[1] != width:
            raise ValueError(
                f"{what} must be shape (count, {width}), got {a.shape}"
            )
        return a, False
    raise ValueError(f"{what} must be 1-D or 2-D, got {a.ndim}-D")


def _training_kwargs(lr, lr_min_frac, lr_decay_epochs, restore_best, beta1, beta2, eps):
    return dict(lr=lr, lr_min_frac=lr_min_frac, lr_decay_epochs=lr_decay_epochs,
                restore_best=restore_best, beta1=beta1, beta2=beta2, eps=eps)


def paint_stripes(x, size: int) -> np.ndarray:
    """Lay a short vector onto a field as contiguous stripes.

    Cell j of the result takes ``x[j * len(x) // size]``, so value i of
    ``x`` fills a block of about ``size / len(x)`` cells and every cell is
    written. This is how a state vector or an action vector that is much
    shorter than the cube becomes a field the encoder can work with.

    Parameters
    ----------
    x : array
        Shape ``(d,)`` or ``(count, d)`` with ``1 <= d <= size``.
    size : int
        Cells in the field: ``N`` for a view, ``code_size`` for an action.

    Returns
    -------
    ndarray
        float32, shape ``(size,)`` or ``(count, size)``.
    """
    x = _f32(x)
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if x.ndim == 1:
        return _paint_stripes(x, size)
    if x.ndim == 2:
        d = x.shape[1]
        if d < 1 or d > size:
            raise ValueError(f"vector length must be in [1, {size}], got {d}")
        idx = (np.arange(size, dtype=np.int64) * d) // size
        return np.ascontiguousarray(x[:, idx], dtype=np.float32)
    raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")


def mean_abs(x) -> float:
    """Mean of absolute values of a float array."""
    return float(_mean_abs(_f32(x)))


def rms(x) -> float:
    """Root-mean-square of a float array."""
    return float(_rms(_f32(x)))


def _fit_loop(core, count, epochs, batch_size, shuffle_seed, verbose,
              accumulate, evaluate):
    """The batch cycle shared by WorldModel.fit and Decoder.fit.

    accumulate(idx) runs one batch and returns the summed loss; evaluate()
    returns the validation metric or None.
    """
    n_use = count - (count % batch_size)
    if n_use == 0:
        n_use = count
    rng = np.random.default_rng(shuffle_seed)
    for epoch in range(epochs):
        core.set_epoch(epoch, epochs)
        order = rng.permutation(count)
        loss_sum = 0.0
        for start in range(0, n_use, batch_size):
            core.begin_batch()
            loss_sum += accumulate(order[start:start + batch_size])
            core.end_batch()
        mean_loss = loss_sum / n_use
        metric = evaluate()
        observed = mean_loss if metric is None else metric
        if verbose:
            tail = "" if metric is None else f" val={metric:.6f}"
            print(f"epoch={epoch}/{epochs} train_loss={mean_loss:.6f}{tail}")
        core.observe(observed, epoch)
    core.restore_best()


class WorldModel:
    """A world model with frozen encoders and a trained Predictor.

    The view encoder sits on a cube of dimension ``dim`` with
    ``N = 2**dim`` vertices. A view is a field of N values; its code is
    the first ``2**k`` values of the encoder's episode, ``code_size``
    long. The action encoder is the same kind of reservoir on a cube of
    dimension ``k``; an action is a picture of ``code_size`` values and
    its code is the whole output. The Predictor maps a view code and an
    action code to the next view code. Only the Predictor trains.

    Typical lifecycle: construct, :meth:`encode` and
    :meth:`encode_action` every transition once, :meth:`fit` on the code
    triples, then :meth:`predict` or :meth:`rollout`.

    Parameters
    ----------
    dim : int
        View cube dimension, 5 to 24; N = 2**dim. A WorldModel needs at
        least 6 so that k has room.
    k : int
        Code face dimension, at least 5 and strictly less than ``dim``.
        Also the action cube dimension.
    output_scale : float
        Presentation gain on both encoders at Create. Finite, > 0.
        Default 1. Set each encoder independently afterwards.
    encoder_seed, ic_seed : int
        Encoder weight draw and episode start state. Both encoders use
        both; the action encoder differs only in its cube.
    spectral_radius, leak_rate, input_scaling : float
        Reservoir knobs, all finite. spectral_radius > 0; leak_rate in
        (0, 1]; input_scaling any finite value.
    history_depth : int
        Delay line length M, 1 to 64.
    passes : int
        Passes per episode T. 0 means a full tour of whichever cube the
        encoder sits on: N for the view, ``2**k`` for the action.
    z_max : int
        Predictor depth; 0 = use k+1, else >= 2.
    gather_span : int
        Predictor lookback window width, 2 to 6.
    tanh_last : bool
        If True the Predictor's last depth applies tanh.
    seed : int
        Predictor weight draw.
    lr, lr_min_frac, lr_decay_epochs, restore_best, beta1, beta2, eps
        Adam and cosine schedule for the Predictor, as in the sibling
        packages: lr finite > 0; lr_min_frac in [0, 1]; betas in [0, 1);
        eps finite > 0.

    Notes
    -----
    One instance is **not thread-safe** for concurrent calls from
    multiple host threads.

    Every batched method accepts one row (a 1-D array) or many (a 2-D
    array with one row per sample) and returns the same shape.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        *,
        output_scale: float = 1.0,
        encoder_seed: int = 7934791766227647176,
        ic_seed: int = 1,
        spectral_radius: float = 0.999,
        leak_rate: float = 1.0,
        input_scaling: float = 0.02,
        history_depth: int = 8,
        passes: int = 0,
        z_max: int = 0,
        gather_span: int = 3,
        tanh_last: bool = False,
        seed: int = 934791766227647176,
        lr: float = 5e-3,
        lr_min_frac: float = 1.0,
        lr_decay_epochs: int = 0,
        restore_best: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self._ctor = dict(
            dim=dim, k=k,
            encoder_seed=encoder_seed, ic_seed=ic_seed,
            spectral_radius=spectral_radius, leak_rate=leak_rate,
            input_scaling=input_scaling, output_scale=output_scale,
            history_depth=history_depth,
            passes=passes, z_max=z_max, gather_span=gather_span,
            tanh_last=tanh_last, seed=seed,
            **_training_kwargs(lr, lr_min_frac, lr_decay_epochs, restore_best,
                               beta1, beta2, eps),
        )
        self._core = _WorldModel(**self._ctor)

    @classmethod
    def _wrap(cls, core):
        obj = cls.__new__(cls)
        obj._core = core
        d = dict(core.config())
        d.pop("action_output_scale", None)
        obj._ctor = d
        return obj

    # ── Encode ──

    def encode(self, fields) -> np.ndarray:
        """View codes for one field ``(N,)`` or many ``(count, N)``.

        Returns float32 ``(code_size,)`` or ``(count, code_size)``. Codes
        depend only on the field: the encoder is frozen.
        """
        a, one = _rows(fields, self.N, "fields")
        out = self._core.encode(a)
        return out[0] if one else out

    def last_cube(self) -> np.ndarray:
        """Scaled full view episode behind the most recent :meth:`encode`."""
        return self._core.last_cube()

    def last_raw_cube(self) -> np.ndarray:
        """Unscaled full view episode behind the most recent :meth:`encode`."""
        return self._core.last_raw_cube()

    def last_packed(self) -> np.ndarray:
        """Packed E(x) then E(a) from the most recent predict or accumulate."""
        return self._core.last_packed()

    def encode_action(self, pictures) -> np.ndarray:
        """Action codes for one picture ``(code_size,)`` or many
        ``(count, code_size)``. Same shape out, float32.

        This is a full episode on the action cube, not a lookup. Encode
        each distinct action once and reuse the code.
        """
        a, one = _rows(pictures, self.code_size, "pictures")
        out = self._core.encode_action(a)
        return out[0] if one else out

    # ── Predict ──

    def predict(self, z, za) -> np.ndarray:
        """Predicted next view codes. ``z`` and ``za`` are one code each
        or matching ``(count, code_size)`` arrays; a single ``za`` is
        broadcast against many ``z``."""
        zz, one_z = _rows(z, self.code_size, "z")
        aa, one_a = _rows(za, self.code_size, "za")
        if one_a and not one_z:
            aa = np.broadcast_to(aa, zz.shape)
        elif one_z and not one_a:
            zz = np.broadcast_to(zz, aa.shape)
        out = self._core.predict(_f32(zz), _f32(aa))
        return out[0] if (one_z and one_a) else out

    def rollout(self, z0, actions) -> np.ndarray:
        """Chain :meth:`predict` over a plan.

        ``z0`` is one code ``(code_size,)`` or ``(count, code_size)``;
        ``actions`` is ``(H, code_size)`` or ``(count, H, code_size)`` of
        action codes already through :meth:`encode_action`. Returns
        ``(H + 1, code_size)`` or ``(count, H + 1, code_size)``; row 0 is
        ``z0``.
        """
        c = self.code_size
        zz, one = _rows(z0, c, "z0")
        aa = _f32(actions)
        if aa.ndim == 2:
            if not one:
                raise ValueError("actions must be 3-D when z0 has many rows")
            aa = aa.reshape(1, *aa.shape)
        elif aa.ndim != 3:
            raise ValueError(f"actions must be 2-D or 3-D, got {aa.ndim}-D")
        if aa.shape[2] != c:
            raise ValueError(f"actions last dimension must be {c}, got {aa.shape[2]}")
        if aa.shape[0] != zz.shape[0]:
            raise ValueError(
                f"actions has {aa.shape[0]} rows but z0 has {zz.shape[0]}"
            )
        out = self._core.rollout(zz, aa)
        return out[0] if one else out

    def pack(self, z, za) -> np.ndarray:
        """What the Predictor sees: E(x) then E(a), ``2 * code_size``
        long. Rarely needed."""
        zz, one_z = _rows(z, self.code_size, "z")
        aa, one_a = _rows(za, self.code_size, "za")
        out = self._core.pack(zz, aa)
        return out[0] if (one_z and one_a) else out

    # ── Training (one-shot) ──

    def fit(
        self,
        z,
        za,
        z_next,
        *,
        epochs: int,
        batch_size: int = 32,
        val=None,
        shuffle_seed: int = 1,
        verbose: bool = False,
    ) -> "WorldModel":
        """Train the Predictor on code triples with the standard cycle.

        Each epoch shuffles, then per batch runs begin_batch, accumulate
        over the batch, end_batch. With ``restore_best`` the observed
        metric is the mean validation squared error when ``val`` is
        given, else the mean training loss, and the best weights are
        restored at the end. Calling fit again continues from the current
        weights; call :meth:`reset_training` first for a fresh optimizer.

        Parameters
        ----------
        z, za, z_next : ndarray
            Shape ``(count, code_size)`` each: the view code, the action
            code, and the next view code of every transition.
        epochs : int
        batch_size : int
            Pairs per Adam step. A short tail that cannot fill a batch is
            dropped each epoch.
        val : tuple, optional
            ``(z, za, z_next)`` held-out triples; scored with
            :meth:`evaluate` each epoch and fed to observe.
        shuffle_seed : int
        verbose : bool
            Print one line per epoch.
        """
        c = self.code_size
        z, za, z_next = (_f32(x) for x in (z, za, z_next))
        for name, a in (("z", z), ("za", za), ("z_next", z_next)):
            if a.ndim != 2 or a.shape[1] != c:
                raise ValueError(f"{name} must be shape (count, {c}), got {a.shape}")
        if not (z.shape[0] == za.shape[0] == z_next.shape[0]):
            raise ValueError("z, za, z_next must have the same number of rows")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        def accumulate(idx):
            return self._core.accumulate(z[idx], za[idx], z_next[idx])

        def evaluate():
            return None if val is None else self.evaluate(*val)

        _fit_loop(self._core, z.shape[0], epochs, batch_size, shuffle_seed,
                  verbose, accumulate, evaluate)
        return self

    def evaluate(self, z, za, z_next) -> float:
        """Mean squared error per code value of :meth:`predict` against
        ``z_next``, over every row. Lower is better."""
        hat = self.predict(z, za)
        target = _f32(z_next)
        return float(np.mean((hat - target) ** 2))

    # ── Training (custom loop) ──

    def begin_batch(self) -> None:
        """Clear the accumulated gradient. Call at the start of each batch."""
        self._core.begin_batch()

    def accumulate(self, z, za, z_next) -> float:
        """Forward, loss, backward for one triple or many. Returns the
        summed loss, 0.5 x SSE over the code per pair."""
        zz, _ = _rows(z, self.code_size, "z")
        aa, _ = _rows(za, self.code_size, "za")
        nn, _ = _rows(z_next, self.code_size, "z_next")
        return float(self._core.accumulate(zz, aa, nn))

    def end_batch(self) -> None:
        """One Adam step on the accumulated gradient."""
        self._core.end_batch()

    def set_epoch(self, epoch: int, num_epochs: int = 0) -> None:
        """Apply the cosine learning-rate schedule for this epoch."""
        self._core.set_epoch(epoch, num_epochs)

    def observe(self, metric: float, epoch: int) -> None:
        """Snapshot the Predictor weights on a new low metric (restore_best only)."""
        self._core.observe(float(metric), int(epoch))

    def restore_best(self) -> None:
        """Write the best-metric snapshot back into the Predictor."""
        self._core.restore_best()

    def reset_training(self) -> None:
        """Forget the optimizer run: Adam moments, step count, lr, best
        snapshot. Weights are untouched."""
        self._core.reset_training()

    # ── Weights / gradient ──

    @property
    def weights(self) -> np.ndarray:
        """Copy of the Predictor weights: depth, axis, tap, vertex."""
        return self._core.weights()

    @weights.setter
    def weights(self, w) -> None:
        w = _f32(np.ravel(w))
        if w.size != self.num_weights:
            raise ValueError(
                f"weights length ({w.size}) must equal num_weights ({self.num_weights})"
            )
        self._core.load_weights(w)

    @property
    def grad(self) -> np.ndarray:
        """Copy of the accumulated gradient, same layout as :attr:`weights`."""
        return self._core.grad()

    def add_grad(self, g) -> None:
        """Sum a gradient of the same layout onto the accumulated one
        (for hosts that train replicas)."""
        g = _f32(np.ravel(g))
        if g.size != self.num_weights:
            raise ValueError(
                f"grad length ({g.size}) must equal num_weights ({self.num_weights})"
            )
        self._core.add_grad(g)

    # ── Properties ──

    @property
    def dim(self) -> int:
        """View cube dimension."""
        return int(self._core.dim)

    @property
    def k(self) -> int:
        """Code face dimension; also the action cube dimension."""
        return int(self._core.k)

    @property
    def N(self) -> int:
        """View field length, 2**dim."""
        return int(self._core.field_size)

    @property
    def code_size(self) -> int:
        """Code length, 2**k; also the action picture length."""
        return int(self._core.code_size)

    @property
    def passes(self) -> int:
        """Resolved passes per view episode (0 in the constructor becomes N)."""
        return int(self._core.passes)

    @property
    def action_passes(self) -> int:
        """Resolved passes per action episode (0 becomes 2**k)."""
        return int(self._core.action_passes)

    @property
    def z_max(self) -> int:
        """Resolved Predictor depth (0 in the constructor becomes k+1)."""
        return int(self._core.config()["z_max"])

    @property
    def view_output_scale(self) -> float:
        return float(self._core.view_output_scale())

    @property
    def action_output_scale(self) -> float:
        return float(self._core.action_output_scale())

    def set_view_output_scale(self, scale: float) -> None:
        self._core.set_view_output_scale(float(scale))
        self._ctor["output_scale"] = float(self._core.view_output_scale())

    def set_action_output_scale(self, scale: float) -> None:
        self._core.set_action_output_scale(float(scale))

    def suggest_view_output_scale(self, z, target_rms: float = 1.0) -> float:
        """Scale that would bring already-run raw view codes to RMS target."""
        return float(self._core.suggest_view_output_scale(_f32(z), float(target_rms)))

    def suggest_action_output_scale(self, za, target_rms: float = 1.0) -> float:
        """Scale that would bring already-run raw action codes to RMS target."""
        return float(self._core.suggest_action_output_scale(_f32(za), float(target_rms)))

    def fit_view_output_scale(self, z, target_rms: float = 1.0) -> None:
        self._core.fit_view_output_scale(_f32(z), float(target_rms))
        self._ctor["output_scale"] = float(self._core.view_output_scale())

    def fit_action_output_scale(self, za, target_rms: float = 1.0) -> None:
        self._core.fit_action_output_scale(_f32(za), float(target_rms))

    @property
    def num_weights(self) -> int:
        """Predictor weights: 2**(k+1) * (k+1) * gather_span * z_max."""
        return int(self._core.num_weights)

    @property
    def realized_spectral_radius(self) -> float:
        """Estimated spectral radius of the view encoder's recurrent block."""
        return float(self._core.realized_spectral_radius)

    @property
    def action_realized_spectral_radius(self) -> float:
        """Estimated spectral radius of the action encoder's recurrent block."""
        return float(self._core.action_realized_spectral_radius)

    def __repr__(self) -> str:
        return (
            f"WorldModel(dim={self.dim}, k={self.k}, N={self.N}, "
            f"code_size={self.code_size}, z_max={self.z_max}, "
            f"weights={self.num_weights})"
        )

    # ── Persistence ──

    def save(self, path) -> None:
        """Write the config and the Predictor weights to a file the C++
        WorldModel::Load also reads. Optimizer state is not saved."""
        self._core.save(str(pathlib.Path(path)))

    @classmethod
    def load(cls, path) -> "WorldModel":
        """Read a model written by :meth:`save` or by C++ WorldModel::Save."""
        return cls._wrap(_WorldModel.load(str(pathlib.Path(path))))

    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        return {
            "_version": self._PERSISTENCE_VERSION,
            "ctor": dict(self._ctor),
            "action_output_scale": self.action_output_scale,
            "weights": self._core.weights(),
        }

    def __setstate__(self, state: dict) -> None:
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Model was saved with persistence version {version}, but this "
                f"version only supports up to {self._PERSISTENCE_VERSION}. "
                f"Upgrade hypercube-worldmodel."
            )
        ctor = dict(state["ctor"])
        ctor.pop("action_scale", None)
        ctor.pop("action_output_scale", None)
        self.__init__(**ctor)
        if "action_output_scale" in state:
            self.set_action_output_scale(state["action_output_scale"])
        self._core.load_weights(_f32(state["weights"]))


class Decoder:
    """Reconstruction: a locally connected net that turns a code back
    into a field.

    The Decoder places a code of ``2**k`` values on the first ``2**k``
    vertices of a cube of dimension ``dim``, feeds every other vertex
    zero, and writes a field of ``N = 2**dim`` values. It trains on
    pairs of a code and the field it came from. Pair it with a
    :class:`WorldModel` by giving it the same ``dim`` and ``k``.

    Parameters
    ----------
    dim : int
        Output cube dimension; the encoder's ``dim``. 4 to 24.
    k : int
        Input face dimension; the WorldModel's ``k``. Strictly less
        than ``dim``.
    z_max : int
        Depth; 0 = use dim, else >= 2.
    gather_span : int
        Lookback window width, 2 to 6.
    tanh_last : bool
        If True the last depth applies tanh.
    seed : int
        Weight draw.
    lr, lr_min_frac, lr_decay_epochs, restore_best, beta1, beta2, eps
        Adam and cosine schedule, as for :class:`WorldModel`.

    Notes
    -----
    Call :meth:`fit_input_scale` on the training codes before the first
    epoch; :meth:`fit` does it for you. One instance is **not
    thread-safe** for concurrent calls.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        *,
        z_max: int = 0,
        gather_span: int = 2,
        tanh_last: bool = False,
        seed: int = 934791766227647176,
        lr: float = 5e-3,
        lr_min_frac: float = 1.0,
        lr_decay_epochs: int = 0,
        restore_best: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self._ctor = dict(
            dim=dim, k=k, z_max=z_max, gather_span=gather_span,
            tanh_last=tanh_last, seed=seed,
            **_training_kwargs(lr, lr_min_frac, lr_decay_epochs, restore_best,
                               beta1, beta2, eps),
        )
        self._core = _Decoder(**self._ctor)

    @classmethod
    def _wrap(cls, core):
        obj = cls.__new__(cls)
        obj._core = core
        obj._ctor = dict(core.config())
        return obj

    # ── Inference ──

    def decode(self, codes) -> np.ndarray:
        """Reconstructed fields for one code ``(code_size,)`` or many
        ``(count, code_size)``. Returns ``(N,)`` or ``(count, N)``."""
        a, one = _rows(codes, self.code_size, "codes")
        out = self._core.decode(a)
        return out[0] if one else out

    # ── Training (one-shot) ──

    def fit(
        self,
        codes,
        fields,
        *,
        epochs: int,
        batch_size: int = 32,
        val=None,
        shuffle_seed: int = 1,
        verbose: bool = False,
        fit_input_scale: bool = True,
    ) -> "Decoder":
        """Train on code, field pairs with the standard cycle.

        Sets the input scale from ``codes`` first unless
        ``fit_input_scale`` is False (for a second fit, or a scale set by
        hand). With ``restore_best`` the observed metric is the mean
        validation squared error when ``val`` is ``(codes, fields)``,
        else the mean training loss.
        """
        c, n = self.code_size, self.N
        codes = _f32(codes)
        fields = _f32(fields)
        if codes.ndim != 2 or codes.shape[1] != c:
            raise ValueError(f"codes must be shape (count, {c}), got {codes.shape}")
        if fields.ndim != 2 or fields.shape[1] != n:
            raise ValueError(f"fields must be shape (count, {n}), got {fields.shape}")
        if codes.shape[0] != fields.shape[0]:
            raise ValueError("codes and fields must have the same number of rows")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if fit_input_scale:
            self.fit_input_scale(codes)

        def accumulate(idx):
            return self._core.accumulate(codes[idx], fields[idx])

        def evaluate():
            return None if val is None else self.evaluate(*val)

        _fit_loop(self._core, codes.shape[0], epochs, batch_size, shuffle_seed,
                  verbose, accumulate, evaluate)
        return self

    def evaluate(self, codes, fields) -> float:
        """Mean squared error per field value of :meth:`decode` against
        ``fields``, over every row."""
        out = self.decode(codes)
        return float(np.mean((out - _f32(fields)) ** 2))

    # ── Training (custom loop) ──

    def begin_batch(self) -> None:
        """Clear the accumulated gradient. Call at the start of each batch."""
        self._core.begin_batch()

    def accumulate(self, codes, fields) -> float:
        """Forward, loss, backward for one pair or many. Returns the
        summed loss, 0.5 x SSE over the field per pair."""
        cc, _ = _rows(codes, self.code_size, "codes")
        ff, _ = _rows(fields, self.N, "fields")
        return float(self._core.accumulate(cc, ff))

    def end_batch(self) -> None:
        """One Adam step on the accumulated gradient."""
        self._core.end_batch()

    def set_epoch(self, epoch: int, num_epochs: int = 0) -> None:
        """Apply the cosine learning-rate schedule for this epoch."""
        self._core.set_epoch(epoch, num_epochs)

    def observe(self, metric: float, epoch: int) -> None:
        """Snapshot the weights on a new low metric (restore_best only)."""
        self._core.observe(float(metric), int(epoch))

    def restore_best(self) -> None:
        """Write the best-metric snapshot back."""
        self._core.restore_best()

    def reset_training(self) -> None:
        """Forget the optimizer run. Weights and input scale are untouched."""
        self._core.reset_training()

    # ── Input scale ──

    def fit_input_scale(self, codes) -> None:
        """Set the input scale to 1 / max |x| over every value in
        ``codes``, so scaled codes lie in [-1, 1]. Frozen after that and
        saved with the weights."""
        self._core.fit_input_scale(_f32(np.ravel(codes)))

    @property
    def input_scale(self) -> float:
        return float(self._core.input_scale)

    @input_scale.setter
    def input_scale(self, scale: float) -> None:
        self._core.set_input_scale(float(scale))

    # ── Weights / gradient ──

    @property
    def weights(self) -> np.ndarray:
        """Copy of the weights: depth, axis, tap, vertex."""
        return self._core.weights()

    @weights.setter
    def weights(self, w) -> None:
        w = _f32(np.ravel(w))
        if w.size != self.num_weights:
            raise ValueError(
                f"weights length ({w.size}) must equal num_weights ({self.num_weights})"
            )
        self._core.load_weights(w)

    @property
    def grad(self) -> np.ndarray:
        """Copy of the accumulated gradient, same layout as :attr:`weights`."""
        return self._core.grad()

    def add_grad(self, g) -> None:
        """Sum a gradient of the same layout onto the accumulated one."""
        g = _f32(np.ravel(g))
        if g.size != self.num_weights:
            raise ValueError(
                f"grad length ({g.size}) must equal num_weights ({self.num_weights})"
            )
        self._core.add_grad(g)

    # ── Properties ──

    @property
    def dim(self) -> int:
        return int(self._ctor["dim"])

    @property
    def k(self) -> int:
        return int(self._ctor["k"])

    @property
    def N(self) -> int:
        """Field length, 2**dim."""
        return int(self._core.field_size)

    @property
    def code_size(self) -> int:
        """Code length, 2**k."""
        return int(self._core.code_size)

    @property
    def z_max(self) -> int:
        """Resolved depth (0 in the constructor becomes dim)."""
        return int(self._core.config()["z_max"])

    @property
    def num_weights(self) -> int:
        """N * dim * gather_span * z_max."""
        return int(self._core.num_weights)

    def __repr__(self) -> str:
        return (
            f"Decoder(dim={self.dim}, k={self.k}, N={self.N}, "
            f"code_size={self.code_size}, z_max={self.z_max}, "
            f"weights={self.num_weights})"
        )

    # ── Persistence ──

    def save(self, path) -> None:
        """Write the config, the input scale, and the weights to a file
        the C++ Decoder::Load also reads. Optimizer state is not saved."""
        self._core.save(str(pathlib.Path(path)))

    @classmethod
    def load(cls, path) -> "Decoder":
        """Read a decoder written by :meth:`save` or by C++ Decoder::Save."""
        return cls._wrap(_Decoder.load(str(pathlib.Path(path))))

    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        return {
            "_version": self._PERSISTENCE_VERSION,
            "ctor": dict(self._ctor),
            "input_scale": self.input_scale,
            "weights": self._core.weights(),
        }

    def __setstate__(self, state: dict) -> None:
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Decoder was saved with persistence version {version}, but this "
                f"version only supports up to {self._PERSISTENCE_VERSION}. "
                f"Upgrade hypercube-worldmodel."
            )
        self.__init__(**dict(state["ctor"]))
        self._core.set_input_scale(float(state["input_scale"]))
        self._core.load_weights(_f32(state["weights"]))


def min_dim(obs_dim: int) -> int:
    """Floor view-cube dim for a vector of this length. Not a config."""
    return int(_min_dim(int(obs_dim)))


def min_k(act_dim: int) -> int:
    """Floor action-cube dim for a vector of this length. Not a config."""
    return int(_min_k(int(act_dim)))


def no_change_mse(z, zn) -> float:
    """MSE of predicting z_{t+1} = z_t."""
    return float(_no_change_mse(_f32(z), _f32(zn)))


def one_step_ratio(mse, z, zn) -> float:
    """mse / no-change MSE. Lower is better."""
    return float(_one_step_ratio(float(mse), _f32(z), _f32(zn)))


def action_sensitivity(z, predict, encode_action, act_dim, mse, rng, n=4000):
    """Same z, two random actions in [-1, 1]. Returns div, mse, act (div/mse)."""
    n = min(int(n), len(z))
    zb = z[:n]
    za1 = encode_action(rng.uniform(-1, 1, (n, act_dim)).astype(np.float32))
    za2 = encode_action(rng.uniform(-1, 1, (n, act_dim)).astype(np.float32))
    return _action_sensitivity_from_preds(_f32(predict(zb, za1)), _f32(predict(zb, za2)),
                                          float(mse))


def lin_r2(z, y, z_val, y_val, ridge=1e-4):
    """Per-dimension val R² of linear ridge from z to y. Returns min, mean, r2."""
    y = np.asarray(y, np.float32)
    yv = np.asarray(y_val, np.float32)
    if y.ndim == 1:
        y = y[:, None]
        yv = yv[:, None]
    return _lin_r2(_f32(z), _f32(y), _f32(z_val), _f32(yv), float(ridge))


def rollout_error(model, obs_ep, act_ep, H: int, windows: int, rng):
    """obs_ep (E, T+1, obs_dim), act_ep (E, T, act_dim). Returns dict of (H,) arrays."""
    E, T = act_ep.shape[:2]
    e_idx = rng.integers(0, E, windows)
    t_idx = rng.integers(0, T - H + 1, windows)
    a = act_ep[e_idx[:, None], t_idx[:, None] + np.arange(H)]
    o = obs_ep[e_idx[:, None], t_idx[:, None] + np.arange(H + 1)]
    W = windows
    z_true = model.encode(o.reshape(W * (H + 1), -1)).reshape(W, H + 1, -1)
    z_pred = model.rollout(z_true[:, 0], a)
    return _rollout_error_from_codes(_f32(z_pred), _f32(z_true))


class ActionSpace:
    """Bounds only. A planner reads .low and .high; not a gymnasium Box."""

    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class Normaliser:
    """(x - mean) / std, clipped to [-clip, clip], then scaled to [-1, 1]."""

    def __init__(self, mean, std, clip: float = 3.0):
        self._core = _Normaliser.from_state(_f32(np.ravel(mean)), _f32(np.ravel(std)),
                                            float(clip))

    @classmethod
    def fit(cls, x, clip: float = 3.0) -> "Normaliser":
        x = _f32(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"Normaliser.fit x must be 1-D or 2-D, got {x.ndim}-D")
        obj = cls.__new__(cls)
        obj._core = _Normaliser.fit(x, float(clip))
        return obj

    @classmethod
    def _wrap(cls, core) -> "Normaliser":
        obj = cls.__new__(cls)
        obj._core = core
        return obj

    def __call__(self, x):
        x = _f32(x)
        one = x.ndim == 1
        if one:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"Normaliser x must be 1-D or 2-D, got {x.ndim}-D")
        out = self._core.apply(x)
        return out[0] if one else out

    @property
    def mean(self):
        return self._core.mean()

    @property
    def std(self):
        return self._core.std()

    @property
    def clip(self) -> float:
        return float(self._core.clip)

    def state(self):
        return {"mean": self.mean, "std": self.std, "clip": self.clip}

    @classmethod
    def from_state(cls, d) -> "Normaliser":
        return cls(d["mean"], d["std"], float(np.asarray(d["clip"])))


class Head:
    """kind is the feature map; sign tells plan_cost whether to negate."""

    def __init__(self, kind: str = "quadratic", ridge: float = 1e-3, sign: str = "cost"):
        self._core = _Head(str(kind), float(ridge), str(sign))

    @classmethod
    def _wrap(cls, core) -> "Head":
        obj = cls.__new__(cls)
        obj._core = core
        return obj

    @property
    def kind(self) -> str:
        return str(self._core.kind)

    @property
    def sign(self) -> str:
        return str(self._core.sign)

    @property
    def ridge(self) -> float:
        return float(self._core.ridge)

    @property
    def uses_za(self) -> bool:
        return bool(self._core.uses_za)

    @property
    def w(self):
        return self._core.w()

    @property
    def b(self) -> float:
        return float(self._core.b)

    @property
    def mu(self):
        return self._core.mu()

    @property
    def sd(self):
        return self._core.sd()

    def fit(self, z, y, za=None) -> "Head":
        z = _f32(z)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        za_a = None if za is None else _f32(za)
        if za_a is not None and za_a.ndim == 1:
            za_a = za_a.reshape(1, -1)
        self._core.fit(z, y, za_a)
        return self

    def __call__(self, z, za=None):
        z = _f32(z)
        one = z.ndim == 1
        if one:
            z = z.reshape(1, -1)
        za_a = None if za is None else _f32(za)
        if za_a is not None and za_a.ndim == 1:
            za_a = za_a.reshape(1, -1)
        out = self._core.apply(z, za_a)
        return out[0] if one else out

    def score(self, z, y, za=None):
        z = _f32(z)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        za_a = None if za is None else _f32(za)
        if za_a is not None and za_a.ndim == 1:
            za_a = za_a.reshape(1, -1)
        return self._core.score(z, y, za_a)

    def plan_cost(self):
        def cost(zs, goal_z=None):
            zs = _f32(zs)
            if zs.ndim != 3:
                raise ValueError("plan_cost zs must be (B, H+1, code)")
            return self._core.plan_cost(zs)

        return cost

    def state(self):
        return {
            "w": self.w, "b": self.b, "mu": self.mu, "sd": self.sd,
            "ridge": self.ridge, "kind": self.kind, "sign": self.sign,
            "uses_za": np.bool_(self.uses_za),
        }

    @classmethod
    def from_state(cls, d) -> "Head":
        kind = str(np.asarray(d["kind"]).item() if "kind" in d else "quadratic")
        sign = str(np.asarray(d["sign"]).item() if "sign" in d else "cost")
        if kind in ("cost", "reward"):
            sign = kind
            kind = "quadratic"
        ridge = float(np.asarray(d["ridge"]))
        uses_za = bool(np.asarray(d["uses_za"]).item()) if "uses_za" in d else False
        w = _f32(np.ravel(d["w"]))
        mu = _f32(np.ravel(d["mu"]))
        sd = _f32(np.ravel(d["sd"]))
        code = int(np.asarray(d["code"]).item()) if "code" in d else 0
        if code == 0:
            # Infer code from feature count: linear nf=c, quadratic nf=c+c(c+1)/2.
            nf = int(w.size)
            if uses_za:
                if nf % 2:
                    raise ValueError("Head.from_state: odd feature count with za")
                nf //= 2
            if kind == "linear":
                code = nf
            else:
                # c + c(c+1)/2 = nf  =>  c(c+3)/2 = nf  => c^2 + 3c - 2 nf = 0
                disc = 9 + 8 * nf
                code = int((-3 + disc ** 0.5) / 2)
        h = cls._wrap(_Head.from_state(kind, sign, ridge, uses_za, code, w,
                                       float(np.asarray(d["b"])), mu, sd))
        return h


class VectorModel:
    """WorldModel plus optional normalisers, PaintStripes, raw-action rollout.

    Always paints with paint_stripes. encode / encode_action take raw vectors.
    rollout takes raw actions. Named Heads and optional action bounds live here.
    """

    def __init__(self, wm, obs_norm=None, act_norm=None,
                 action_low=None, action_high=None, heads=None, cost=None):
        if not isinstance(wm, WorldModel):
            raise TypeError("VectorModel needs a WorldModel")
        self._wm = wm
        self._core = _VectorModel(wm._core)
        self._cost = cost
        if obs_norm is not None:
            self._core.set_obs_norm(obs_norm._core)
        if act_norm is not None:
            self._core.set_act_norm(act_norm._core)
        if action_low is not None and action_high is not None:
            self._core.set_action_bounds(_f32(np.ravel(action_low)),
                                         _f32(np.ravel(action_high)))
        if heads:
            for name, h in dict(heads).items():
                self._core.set_head(str(name), h._core)

    @classmethod
    def _wrap(cls, core) -> "VectorModel":
        obj = cls.__new__(cls)
        obj._core = core
        obj._wm = WorldModel._wrap(core.world())
        obj._cost = None
        return obj

    @property
    def wm(self) -> WorldModel:
        return self._wm

    @property
    def latent_dim(self) -> int:
        return int(self._wm.code_size)

    @property
    def obs_norm(self):
        n = self._core.obs_norm()
        return None if n is None else Normaliser._wrap(n)

    @property
    def act_norm(self):
        n = self._core.act_norm()
        return None if n is None else Normaliser._wrap(n)

    @property
    def action_space(self):
        if not self._core.has_action_bounds():
            return None
        return ActionSpace(self._core.action_low(), self._core.action_high())

    @property
    def heads(self) -> dict:
        return {name: Head._wrap(self._core.get_head(name))
                for name in self._core.head_names()}

    @property
    def obs_dim(self) -> int:
        return int(self._core.obs_dim)

    @property
    def act_dim(self) -> int:
        return int(self._core.act_dim)

    def encode(self, obs):
        obs = _f32(obs)
        one = obs.ndim == 1
        if one:
            obs = obs.reshape(1, -1)
        if obs.ndim != 2:
            raise ValueError(f"obs must be 1-D or 2-D, got {obs.ndim}-D")
        out = self._core.encode(obs)
        return out[0] if one else out

    def encode_action(self, a):
        a = _f32(a)
        one = a.ndim == 1
        if one:
            a = a.reshape(1, -1)
        if a.ndim != 2:
            raise ValueError(f"a must be 1-D or 2-D, got {a.ndim}-D")
        out = self._core.encode_action(a)
        return out[0] if one else out

    def predict(self, z, za):
        zz, one_z = _rows(z, self._wm.code_size, "z")
        aa, one_a = _rows(za, self._wm.code_size, "za")
        if one_a and not one_z:
            aa = np.broadcast_to(aa, zz.shape)
        elif one_z and not one_a:
            zz = np.broadcast_to(zz, aa.shape)
        out = self._core.predict(_f32(zz), _f32(aa))
        return out[0] if (one_z and one_a) else out

    def rollout(self, z0, actions):
        c = self._wm.code_size
        zz, one = _rows(z0, c, "z0")
        aa = _f32(actions)
        if aa.ndim == 2:
            if not one:
                raise ValueError("actions must be 3-D when z0 has many rows")
            aa = aa.reshape(1, *aa.shape)
        elif aa.ndim != 3:
            raise ValueError(f"actions must be 2-D or 3-D, got {aa.ndim}-D")
        out = self._core.rollout(zz, aa)
        return out[0] if one else out

    def cost(self, zs, goal_z=None):
        if self._cost is not None:
            return self._cost(zs, goal_z)
        zs = _f32(zs)
        if zs.ndim != 3:
            raise ValueError("cost zs must be (B, H+1, code)")
        g = None if goal_z is None else _f32(np.ravel(goal_z))
        return self._core.cost(zs, g)

    def set_head(self, name: str, head: Head) -> None:
        self._core.set_head(str(name), head._core)

    def set_obs_dim(self, d: int) -> None:
        self._core.set_obs_dim(int(d))

    def set_act_dim(self, d: int) -> None:
        self._core.set_act_dim(int(d))

    def set_meta(self, key: str, value: str) -> None:
        self._core.set_meta(str(key), str(value))

    def meta(self, key: str) -> str:
        return str(self._core.meta(str(key)))

    def save(self, path) -> None:
        """Write a VectorModel file the C++ VectorModel::Load also reads."""
        self._core.save(str(pathlib.Path(path)))

    @classmethod
    def load(cls, path) -> "VectorModel":
        """Read a file written by :meth:`save` or by C++ VectorModel::Save.
        A bare WorldModel HWM1 file loads as wm-only."""
        return cls._wrap(_VectorModel.load(str(pathlib.Path(path))))

