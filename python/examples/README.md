# Python examples

Demo scripts for the hypercube-worldmodel API. They call only the
installed package.

| Script | What it shows |
|--------|----------------|
| [plane_point.py](plane_point.py) | A point on a plane moved by a bounded velocity: encode, fit with a validation set, a held-out score against the identity guess, rollout, a Decoder beside the model, and a save and load round trip |
| [plan_toy.py](plan_toy.py) | DMC protocol adapter (encode, encode_action, predict, rollout on raw actions, cost, action_space) driving CEM to a goal on the same toy world |

## How to run

These files live in the GitHub repository. They are not part of pip
install; use the [Quick start](../README.md#quick-start) if you only
want a snippet.

From a clone of HypercubeWorldModel (repository root), after installing
the package:

```bash
pip install hypercube-worldmodel
# or from this tree:  pip install ./python
python python/examples/plane_point.py
python python/examples/plan_toy.py
```

## What these are not

- **Not** the C++ programs (the smoke test, quick_start, WorldModelTest,
  TerrainWalkerTest) or the study write-ups. Those live under
  [tests/](../../tests/) and [docs/](../../docs/).
- **Not** automated tests. Package tests are
  [tests/test_basic.py](../tests/test_basic.py).
- **Not** hard tasks. The toy world is easy onboarding so the API is
  obvious; do not cite its numbers as research results. The planner
  adapter is the shape to copy when the environment is a real suite.

## Going further

| Want | See |
|------|-----|
| Full Python API | [docs/Python_SDK.md](../../docs/Python_SDK.md) |
| C++ product guide | [docs/CPP_SDK.md](../../docs/CPP_SDK.md) |
| Package readme | [README.md](../README.md) |
| The WorldModel class and the action path | [docs/world_model.md](../../docs/world_model.md) |
| The Decoder and its file format | [docs/decoder.md](../../docs/decoder.md) |
