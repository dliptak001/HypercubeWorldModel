# Single source of truth for the package version.
# - pyproject.toml reads this via scikit-build-core dynamic metadata
# - hypercube_worldmodel.__version__ imports it
# - bindings.cpp gets the same string at compile time via CMake and
#   static_asserts it against WorldModel::kVersion in WorldModel.h
__version__ = "1.1.0"
