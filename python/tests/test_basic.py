"""Contract tests for the hypercube_worldmodel package.

Small cubes (dim 6, k 5) keep every test fast; the goal is contract
coverage, not scores.
"""

import pickle

import numpy as np
import pytest

import hypercube_worldmodel as hw

DIM = 6
K = 5


def make_wm(**kw):
    args = dict(dim=DIM, k=K, passes=12, leak_rate=0.25, input_scaling=0.8,
                z_max=6, gather_span=3, tanh_last=True,
                lr=0.03, lr_min_frac=0.05)
    args.update(kw)
    return hw.WorldModel(**args)


def make_dec(**kw):
    args = dict(dim=DIM, k=K, z_max=3, gather_span=2, lr=0.01)
    args.update(kw)
    return hw.Decoder(**args)


def plane(count, seed=0):
    """A point on a plane moved by a bounded velocity: obs, act, next."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
    nxt = np.clip(obs + 0.1 * act, -1, 1).astype(np.float32)
    return obs, act, nxt


def codes(wm, count, seed=0):
    obs, act, nxt = plane(count, seed)
    z = wm.encode(hw.paint_stripes(obs, wm.N))
    za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
    zn = wm.encode(hw.paint_stripes(nxt, wm.N))
    return z, za, zn


# ── Version ──

def test_version():
    assert isinstance(hw.__version__, str) and hw.__version__
    assert hw.__version__ == hw._core.__version__
    assert hw.__version__ == hw._core.cpp_version


# ── mean_abs / rms ──

def test_mean_abs_and_rms():
    x = np.array([-0.2, 0.2], dtype=np.float32)
    assert hw.mean_abs(x) == pytest.approx(0.2)
    assert hw.rms(x) == pytest.approx(0.2)
    assert hw.mean_abs(np.zeros(0, dtype=np.float32)) == 0.0
    assert hw.rms(np.zeros(0, dtype=np.float32)) == 0.0


# ── paint_stripes ──

def test_paint_stripes_shapes_and_values():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = hw.paint_stripes(x, 9)
    assert out.shape == (9,) and out.dtype == np.float32
    np.testing.assert_array_equal(out, [1, 1, 1, 2, 2, 2, 3, 3, 3])
    both = hw.paint_stripes(np.stack([x, 2 * x]), 9)
    assert both.shape == (2, 9)
    np.testing.assert_array_equal(both[0], out)
    np.testing.assert_array_equal(both[1], 2 * out)


def test_paint_stripes_matches_cpp_for_uneven_blocks():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(7).astype(np.float32)
    one = hw.paint_stripes(x, 64)                 # C++ path
    two = hw.paint_stripes(x[None, :], 64)[0]     # NumPy path
    np.testing.assert_array_equal(one, two)


def test_paint_stripes_rejects_bad_lengths():
    with pytest.raises(ValueError):
        hw.paint_stripes(np.zeros(0, dtype=np.float32), 8)
    with pytest.raises(ValueError):
        hw.paint_stripes(np.zeros(9, dtype=np.float32), 8)
    with pytest.raises(ValueError):
        hw.paint_stripes(np.zeros((2, 9), dtype=np.float32), 8)


# ── Construction / validation ──

def test_sizes():
    wm = make_wm()
    assert wm.dim == DIM and wm.k == K
    assert wm.N == 2 ** DIM and wm.code_size == 2 ** K
    assert wm.z_max == 6
    assert wm.num_weights == 2 ** (K + 1) * (K + 1) * 3 * 6
    assert wm.passes == 12 and wm.action_passes == 12


def test_passes_zero_resolves_per_cube():
    wm = make_wm(passes=0)
    assert wm.passes == 2 ** DIM
    assert wm.action_passes == 2 ** K


def test_z_max_zero_resolves_to_k_plus_one():
    assert make_wm(z_max=0).z_max == K + 1


def test_k_bounds():
    with pytest.raises(ValueError):
        hw.WorldModel(dim=5, k=5)
    with pytest.raises(ValueError):
        hw.WorldModel(dim=DIM, k=DIM)
    with pytest.raises(ValueError):
        hw.WorldModel(dim=DIM, k=4)


def test_bad_config_throws():
    for bad in (dict(output_scale=0.0), dict(output_scale=float("nan")),
                dict(gather_span=1), dict(gather_span=7), dict(z_max=1),
                dict(leak_rate=0.0), dict(leak_rate=float("inf")),
                dict(spectral_radius=float("nan")), dict(input_scaling=float("nan")),
                dict(history_depth=0), dict(lr=-1.0), dict(lr=float("nan")),
                dict(lr_min_frac=2.0), dict(beta1=1.0), dict(eps=0.0)):
        with pytest.raises(ValueError):
            make_wm(**bad)


# ── Encode ──

def test_encode_shapes_and_determinism():
    wm1, wm2 = make_wm(), make_wm()
    field = np.linspace(-1, 1, wm1.N, dtype=np.float32)
    z1, z2 = wm1.encode(field), wm2.encode(field)
    assert z1.shape == (wm1.code_size,) and z1.dtype == np.float32
    np.testing.assert_array_equal(z1, z2)
    many = wm1.encode(np.stack([field, -field]))
    assert many.shape == (2, wm1.code_size)
    np.testing.assert_array_equal(many[0], z1)


def test_encode_is_first_face_of_last_cube():
    wm = make_wm()
    field = np.linspace(-1, 1, wm.N, dtype=np.float32)
    z = wm.encode(field)
    cube = wm.last_cube()
    assert cube.shape == (wm.N,)
    np.testing.assert_array_equal(cube[: wm.code_size], z)


def test_encode_does_not_keep_state():
    wm = make_wm()
    a = np.linspace(-1, 1, wm.N, dtype=np.float32)
    b = np.cos(np.arange(wm.N, dtype=np.float32))
    za_first = wm.encode(a)
    wm.encode(b)
    np.testing.assert_array_equal(wm.encode(a), za_first)


def test_encode_action_shapes():
    wm = make_wm()
    pic = np.ones(wm.code_size, dtype=np.float32)
    za = wm.encode_action(pic)
    assert za.shape == (wm.code_size,)
    assert wm.encode_action(np.stack([pic, 0.5 * pic])).shape == (2, wm.code_size)
    assert not np.array_equal(za, wm.encode_action(0.5 * pic))


def test_encode_wrong_length_throws():
    wm = make_wm()
    with pytest.raises(ValueError):
        wm.encode(np.zeros(wm.N + 1, dtype=np.float32))
    with pytest.raises(ValueError):
        wm.encode_action(np.zeros(wm.N, dtype=np.float32))   # N, not code_size
    with pytest.raises(ValueError):
        wm.encode(np.zeros((2, 3, wm.N), dtype=np.float32))


def test_seeds_change_codes():
    field = np.linspace(-1, 1, 2 ** DIM, dtype=np.float32)
    z1 = make_wm(encoder_seed=1).encode(field)
    z2 = make_wm(encoder_seed=2).encode(field)
    z3 = make_wm(encoder_seed=1, ic_seed=9).encode(field)
    assert not np.array_equal(z1, z2)
    assert not np.array_equal(z1, z3)


# ── Predict / rollout / pack ──

def test_predict_shapes_and_broadcast():
    wm = make_wm()
    z, za, _ = codes(wm, 4)
    one = wm.predict(z[0], za[0])
    assert one.shape == (wm.code_size,)
    many = wm.predict(z, za)
    assert many.shape == (4, wm.code_size)
    np.testing.assert_array_equal(many[0], one)
    bcast = wm.predict(z, za[0])
    assert bcast.shape == (4, wm.code_size)
    np.testing.assert_array_equal(bcast[0], one)
    with pytest.raises(ValueError):
        wm.predict(z, za[:3])


def test_predict_reads_the_action():
    wm = make_wm()
    z, za, _ = codes(wm, 2)
    assert not np.array_equal(wm.predict(z[0], za[0]), wm.predict(z[0], za[1]))


def test_rollout_chains_predict():
    wm = make_wm()
    z, za, _ = codes(wm, 3)
    path = wm.rollout(z[0], za)
    assert path.shape == (4, wm.code_size)
    np.testing.assert_array_equal(path[0], z[0])
    np.testing.assert_array_equal(path[1], wm.predict(z[0], za[0]))
    np.testing.assert_array_equal(path[2], wm.predict(path[1], za[1]))
    batched = wm.rollout(z[:2], np.stack([za, za]))
    assert batched.shape == (2, 4, wm.code_size)
    np.testing.assert_array_equal(batched[0], path)
    empty = wm.rollout(z[0], za[:0])
    assert empty.shape == (1, wm.code_size)
    with pytest.raises(ValueError):
        wm.rollout(z[:2], za)


def test_pack_layout():
    wm = make_wm()
    z, za, _ = codes(wm, 1)
    p = wm.pack(z[0], za[0])
    assert p.shape == (2 * wm.code_size,)
    np.testing.assert_array_equal(p[: wm.code_size], z[0])
    np.testing.assert_array_equal(p[wm.code_size:], za[0])


def test_output_scales_independent():
    wm = make_wm()
    wm.set_view_output_scale(2.0)
    wm.set_action_output_scale(3.0)
    assert wm.view_output_scale == 2.0
    assert wm.action_output_scale == 3.0
    z, za, _ = codes(wm, 4)
    s = wm.suggest_view_output_scale(z)
    assert s > 0.0
    wm.fit_view_output_scale(z)
    assert abs(wm.view_output_scale - s) < 1e-5


# ── Training ──

def test_fit_beats_identity():
    wm = make_wm(restore_best=True)
    z, za, zn = codes(wm, 256, seed=0)
    zt, zat, znt = codes(wm, 64, seed=1)
    identity = float(np.mean((zt - znt) ** 2))
    wm.fit(z, za, zn, epochs=40, batch_size=16, val=(zt, zat, znt))
    assert wm.evaluate(zt, zat, znt) < 0.5 * identity


def test_fit_validates_shapes():
    wm = make_wm()
    z, za, zn = codes(wm, 8)
    with pytest.raises(ValueError):
        wm.fit(z, za[:4], zn, epochs=1)
    with pytest.raises(ValueError):
        wm.fit(z, za, zn, epochs=0)
    with pytest.raises(ValueError):
        wm.fit(z, za, zn, epochs=1, batch_size=0)
    with pytest.raises(ValueError):
        wm.fit(z[0], za[0], zn[0], epochs=1)


def test_custom_loop_matches_fit_cycle():
    wm = make_wm()
    z, za, zn = codes(wm, 16)
    wm.begin_batch()
    loss = wm.accumulate(z, za, zn)
    assert loss > 0.0
    g = wm.grad
    assert np.any(g != 0.0)
    wm.begin_batch()
    assert not np.any(wm.grad)
    wm.accumulate(z[0], za[0], zn[0])
    single = wm.grad.copy()
    wm.add_grad(single)
    np.testing.assert_allclose(wm.grad, 2.0 * single, rtol=1e-5)
    w0 = wm.weights.copy()
    wm.end_batch()
    assert not np.array_equal(wm.weights, w0)


def test_restore_best_and_reset():
    wm = make_wm(restore_best=True)
    w0 = wm.weights.copy()
    wm.observe(1.0, 0)
    rng = np.random.default_rng(1)
    wm.weights = rng.standard_normal(wm.num_weights).astype(np.float32)
    wm.observe(2.0, 1)          # worse: must not snapshot
    wm.restore_best()
    np.testing.assert_array_equal(wm.weights, w0)
    wm.reset_training()         # forgets the snapshot; weights stay
    np.testing.assert_array_equal(wm.weights, w0)


def test_weights_roundtrip():
    wm = make_wm()
    w = wm.weights
    assert w.size == wm.num_weights
    w2 = np.arange(w.size, dtype=np.float32) / w.size
    wm.weights = w2
    np.testing.assert_array_equal(wm.weights, w2)
    with pytest.raises(ValueError):
        wm.weights = w2[:-1]


# ── Persistence ──

def test_save_load_roundtrip_with_passes_zero(tmp_path):
    wm = make_wm(passes=0, restore_best=True)
    z, za, zn = codes(wm, 32)
    wm.fit(z, za, zn, epochs=3, batch_size=8)
    path = tmp_path / "model.wm"
    wm.save(path)
    again = hw.WorldModel.load(path)
    assert again.passes == wm.passes and again.action_passes == wm.action_passes
    obs, act, _ = plane(1, seed=7)
    f = hw.paint_stripes(obs[0], wm.N)
    p = hw.paint_stripes(act[0], wm.code_size)
    np.testing.assert_array_equal(again.encode(f), wm.encode(f))
    np.testing.assert_array_equal(again.encode_action(p), wm.encode_action(p))
    np.testing.assert_array_equal(again.predict(z[0], za[0]), wm.predict(z[0], za[0]))
    assert again.z_max == wm.z_max


def test_pickle_roundtrip():
    wm = make_wm(passes=0)
    z, za, _ = codes(wm, 4)
    rng = np.random.default_rng(2)
    wm.weights = rng.standard_normal(wm.num_weights).astype(np.float32)
    again = pickle.loads(pickle.dumps(wm))
    np.testing.assert_array_equal(again.weights, wm.weights)
    np.testing.assert_array_equal(again.predict(z, za), wm.predict(z, za))
    assert again.action_passes == 2 ** K


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        hw.WorldModel.load(tmp_path / "missing.wm")


def test_repr():
    assert "WorldModel(dim=6, k=5" in repr(make_wm())
    assert "Decoder(dim=6, k=5" in repr(make_dec())


# ── Decoder ──

def test_decoder_sizes_and_bounds():
    dec = make_dec()
    assert dec.N == 2 ** DIM and dec.code_size == 2 ** K
    assert dec.z_max == 3
    assert dec.num_weights == 2 ** DIM * DIM * 2 * 3
    with pytest.raises(ValueError):
        hw.Decoder(dim=DIM, k=DIM)
    with pytest.raises(ValueError):
        make_dec(gather_span=9)


def test_decoder_pairs_with_worldmodel():
    wm = make_wm()
    dec = make_dec()
    assert dec.code_size == wm.code_size and dec.N == wm.N
    z, _, _ = codes(wm, 3)
    out = dec.decode(z)
    assert out.shape == (3, wm.N)
    assert dec.decode(z[0]).shape == (wm.N,)


def test_decoder_fit_reconstructs():
    wm = make_wm()
    obs, _, _ = plane(256, seed=0)
    fields = hw.paint_stripes(obs, wm.N)
    z = wm.encode(fields)
    dec = make_dec(restore_best=True)
    before = dec.evaluate(z, fields)
    dec.fit(z, fields, epochs=30, batch_size=16, val=(z, fields))
    after = dec.evaluate(z, fields)
    assert after < 0.1 * before
    assert dec.input_scale > 0.0


def test_decoder_input_scale():
    dec = make_dec()
    codes_ = np.array([[0.5, -2.0] + [0.0] * (dec.code_size - 2)], dtype=np.float32)
    dec.fit_input_scale(codes_)
    assert dec.input_scale == pytest.approx(0.5)
    dec.input_scale = 3.0
    assert dec.input_scale == 3.0
    with pytest.raises(ValueError):
        dec.fit_input_scale(np.zeros(dec.code_size, dtype=np.float32))
    with pytest.raises(ValueError):
        dec.input_scale = 0.0


def test_decoder_custom_loop_and_weights():
    dec = make_dec()
    wm = make_wm()
    obs, _, _ = plane(8)
    fields = hw.paint_stripes(obs, wm.N)
    z = wm.encode(fields)
    dec.fit_input_scale(z)
    dec.begin_batch()
    assert dec.accumulate(z, fields) > 0.0
    assert np.any(dec.grad != 0.0)
    w0 = dec.weights.copy()
    dec.end_batch()
    assert not np.array_equal(dec.weights, w0)
    dec.weights = w0
    np.testing.assert_array_equal(dec.weights, w0)
    with pytest.raises(ValueError):
        dec.weights = w0[:-1]


def test_decoder_save_load_and_pickle(tmp_path):
    wm = make_wm()
    obs, _, _ = plane(16)
    fields = hw.paint_stripes(obs, wm.N)
    z = wm.encode(fields)
    dec = make_dec()
    dec.fit(z, fields, epochs=2, batch_size=8)
    path = tmp_path / "model.dec"
    dec.save(path)
    again = hw.Decoder.load(path)
    assert again.input_scale == dec.input_scale
    np.testing.assert_array_equal(again.decode(z), dec.decode(z))
    pickled = pickle.loads(pickle.dumps(dec))
    assert pickled.input_scale == dec.input_scale
    np.testing.assert_array_equal(pickled.decode(z), dec.decode(z))
