"""A planner adapter in the DeepMind Control Suite shape, on a toy world.

The world is the plane point from plane_point.py with a goal: reach a
target position. The adapter is the DMC latent-world-model protocol:
encode, encode_action, predict, rollout, cost, plus action_space.low /
.high. CEM samples raw actions and calls rollout with those arrays;
the adapter paints and encode_action's the block once, then WorldModel
rollout on the codes. Swap the environment for a state-based suite
task and keep this adapter.

encode_action is a full episode on the action cube. That is the hot
path of a sampler. The adapter does it once per CEM iteration, not
once per predict step.
"""

import numpy as np

import hypercube_worldmodel as hw

DIM, K = 6, 5
STEP = 0.1
HORIZON, SAMPLES, ELITES, ITERS = 5, 128, 16, 4
EPISODE_STEPS = 40
rng = np.random.default_rng(0)


# ── A toy environment ──

class PlaneWorld:
    """State is a position in [-1, 1]^2; the action is a velocity in [-1, 1]^2."""

    action_low = np.array([-1.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0], dtype=np.float32)

    def reset(self):
        self.pos = rng.uniform(-1, 1, 2).astype(np.float32)
        return self.pos.copy()

    def step(self, action):
        self.pos = np.clip(self.pos + STEP * action, -1, 1).astype(np.float32)
        return self.pos.copy()


# ── The model, trained on random transitions ──

wm = hw.WorldModel(dim=DIM, k=K, passes=2 * DIM, leak_rate=0.25, input_scaling=0.8,
                   action_scale=0.33, z_max=3 * K, gather_span=5, tanh_last=True,
                   lr=0.03, lr_min_frac=0.05, restore_best=True)

obs = rng.uniform(-1, 1, (512, 2)).astype(np.float32)
act = rng.uniform(-1, 1, (512, 2)).astype(np.float32)
nxt = np.clip(obs + STEP * act, -1, 1).astype(np.float32)
z = wm.encode(hw.paint_stripes(obs, wm.N))
za = wm.encode_action(hw.paint_stripes(act, wm.code_size))
zn = wm.encode(hw.paint_stripes(nxt, wm.N))
wm.fit(z, za, zn, epochs=150, batch_size=16)


# ── The adapter: DMC latent world model protocol ──

class ActionSpace:
    """Bounds only. CEM reads .low and .high; not a gymnasium Box."""

    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class Adapter:
    """WorldModel behind the DMC planner protocol.

    encode and encode_action take raw obs / actions. predict takes codes.
    rollout takes raw actions (B, H, act_dim), paints and encode_action's
    that block once, then WorldModel.rollout on the codes. cost scores a
    rollout against a goal code. The model never sees action bounds.
    """

    def __init__(self, wm, action_low, action_high):
        self.wm = wm
        self.latent_dim = wm.code_size
        self.action_space = ActionSpace(action_low, action_high)

    def encode(self, obs):
        return self.wm.encode(hw.paint_stripes(obs, self.wm.N))

    def encode_action(self, a):
        return self.wm.encode_action(hw.paint_stripes(a, self.wm.code_size))

    def predict(self, z, za):
        return self.wm.predict(z, za)

    def rollout(self, z0, actions):
        # (B, H, act_dim) raw actions -> (B, H+1, latent). One plan is (H, act_dim).
        a = np.asarray(actions, dtype=np.float32)
        z = np.asarray(z0, dtype=np.float32)
        one = a.ndim == 2
        if one:
            a = a[None, ...]
            if z.ndim == 1:
                z = z[None, :]
        B, H, d = a.shape
        codes = self.encode_action(a.reshape(B * H, d)).reshape(B, H, -1)
        out = self.wm.rollout(z, codes)
        return out[0] if one else out

    def cost(self, zs, goal_z):
        # distance between the final code and the goal code; lower is better
        return np.sum((zs[:, -1, :] - goal_z) ** 2, axis=1)


# ── Cross-entropy method with a warm start (DMC guide shape) ──

def cem_plan(model, z0, goal_z, mean=None):
    lo, hi = model.action_space.low, model.action_space.high
    d = lo.shape[0]
    mean = np.zeros((HORIZON, d), dtype=np.float32) if mean is None else mean
    std = 0.5 * (hi - lo) * np.ones((HORIZON, d), dtype=np.float32)
    for _ in range(ITERS):
        acts = np.clip(mean + std * rng.standard_normal((SAMPLES, HORIZON, d)), lo, hi)
        acts = acts.astype(np.float32)
        zs = model.rollout(np.repeat(z0[None], SAMPLES, 0), acts)
        c = model.cost(zs, goal_z)
        elite = acts[np.argsort(c)[:ELITES]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean


# ── Run one episode: plan, execute the first action, replan ──

env = PlaneWorld()
model = Adapter(wm, PlaneWorld.action_low, PlaneWorld.action_high)
goal = np.array([0.6, -0.4], dtype=np.float32)
goal_z = model.encode(goal)

obs = env.reset()
start_dist = float(np.linalg.norm(obs - goal))
mean = None
for t in range(EPISODE_STEPS):
    z0 = model.encode(obs)
    mean = cem_plan(model, z0, goal_z, mean)
    obs = env.step(mean[0])
    mean = np.roll(mean, -1, 0)          # warm start: shift the plan by one step
    mean[-1] = 0.0
end_dist = float(np.linalg.norm(obs - goal))
print(f"distance to goal: start {start_dist:.3f}  end {end_dist:.3f}")
print("reached" if end_dist < 0.15 else "not reached")
