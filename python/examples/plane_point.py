"""Smallest end-to-end hypercube_worldmodel run.

A point on a plane moves by a bounded velocity. The view is the
position, the action is the velocity. Short vectors go through
paint_stripes so the encoder has something to respond to. Prints the
training loss falling, a held-out score against the identity guess (the
current code used as the prediction), a Decoder's reconstruction error,
and a save and load round trip.
"""

import numpy as np

import hypercube_worldmodel as hw

DIM, K = 6, 5
STEP = 0.1
TRAIN, VAL, TEST = 512, 128, 128
EPOCHS, BATCH = 200, 16
rng = np.random.default_rng(0)


def draw(count):
    obs = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
    nxt = np.clip(obs + STEP * act, -1, 1).astype(np.float32)
    return obs, act, nxt


wm = hw.WorldModel(dim=DIM, k=K, passes=2 * DIM, leak_rate=0.25, input_scaling=0.8,
                   z_max=3 * K, gather_span=5, tanh_last=True,
                   lr=0.03, lr_min_frac=0.05, restore_best=True)
print(wm)


def encode(split):
    obs, act, nxt = split
    z = wm.encode(hw.paint_stripes(obs, wm.N))
    za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
    zn = wm.encode(hw.paint_stripes(nxt, wm.N))
    return z, za, zn


train, val, test = draw(TRAIN), draw(VAL), draw(TEST)
z, za, zn = encode(train)
zv, zav, znv = encode(val)
zt, zat, znt = encode(test)

wm.fit(z, za, zn, epochs=EPOCHS, batch_size=BATCH, val=(zv, zav, znv), verbose=False)

model = wm.evaluate(zt, zat, znt)
identity = float(np.mean((zt - znt) ** 2))
print(f"test mse: model {model:.6f}  identity {identity:.6f}  ratio {model / identity:.3f}")

# Rollout: three action codes in, four view codes out; row 0 is the start.
path = wm.rollout(zt[0], zat[:3])
print("rollout shape", path.shape)

# A Decoder beside the model: code in, field back.
fields = hw.paint_stripes(train[0], wm.N)
dec = hw.Decoder(dim=DIM, k=K, gather_span=3, lr=0.01, lr_min_frac=0.05, restore_best=True)
dec.fit(z, fields, epochs=60, batch_size=BATCH)
test_fields = hw.paint_stripes(test[0], wm.N)
print(f"decode mse {dec.evaluate(zt, test_fields):.6f}  field variance {float(np.var(test_fields)):.6f}")
rebuilt = dec.decode(wm.predict(zt[0], zat[0]))   # the predicted next view, as a field
print("rebuilt next view shape", rebuilt.shape)

# Save and load: the same files the C++ classes read.
wm.save("plane_point.wm")
dec.save("plane_point.dec")
again = hw.WorldModel.load("plane_point.wm")
same = np.array_equal(again.predict(zt[0], zat[0]), wm.predict(zt[0], zat[0]))
print("reload predicts the same:", same)
