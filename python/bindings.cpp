// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// Thin pybind11 surface for HypercubeWorldModel. Every call that walks a
// batch does the loop here with the GIL released; shape ergonomics,
// fit, pickle, and docs live in hypercube_worldmodel/__init__.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "Decoder.h"
#include "WorldModel.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

#ifndef HYPERCUBE_WORLDMODEL_VERSION
#  error "HYPERCUBE_WORLDMODEL_VERSION must be set by CMake from hypercube_worldmodel/_version.py"
#endif
static_assert(std::string_view(HYPERCUBE_WORLDMODEL_VERSION) == std::string_view(WorldModel::kVersion),
              "python/hypercube_worldmodel/_version.py and WorldModel::kVersion disagree");

namespace {

// A 1-D or 2-D float32 array seen as rows of a fixed width.
struct Rows
{
    const float* data;
    size_t count;
};

Rows AsRows(const FloatArray& a, size_t width, const char* what)
{
    const auto b = a.request();
    size_t rows = 0, w = 0;
    if (b.ndim == 1)
    {
        rows = 1;
        w = static_cast<size_t>(b.shape[0]);
    }
    else if (b.ndim == 2)
    {
        rows = static_cast<size_t>(b.shape[0]);
        w = static_cast<size_t>(b.shape[1]);
    }
    else
    {
        throw std::invalid_argument(std::string(what) + " must be 1-D or 2-D");
    }
    if (w != width)
        throw std::invalid_argument(std::string(what) + " last dimension must be "
                                    + std::to_string(width) + ", got " + std::to_string(w));
    return {static_cast<const float*>(b.ptr), rows};
}

py::array_t<float> Matrix(size_t rows, size_t width)
{
    return py::array_t<float>({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(width)});
}

py::array_t<float> VectorToArray(const std::vector<float>& v)
{
    py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
    if (!v.empty())
        std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
    return arr;
}

py::array_t<float> PointerToArray(const float* p, size_t n)
{
    py::array_t<float> arr(static_cast<py::ssize_t>(n));
    std::memcpy(arr.mutable_data(), p, n * sizeof(float));
    return arr;
}

void RequireSameRows(size_t a, size_t b, const char* what)
{
    if (a != b)
        throw std::invalid_argument(std::string(what) + ": row counts differ ("
                                    + std::to_string(a) + " vs " + std::to_string(b) + ")");
}

py::dict TrainingDict(const LCNTrainingConfig& t)
{
    py::dict d;
    d["lr"] = t.lr;
    d["lr_min_frac"] = t.lr_min_frac;
    d["lr_decay_epochs"] = t.lr_decay_epochs;
    d["restore_best"] = t.restore_best;
    d["beta1"] = t.beta1;
    d["beta2"] = t.beta2;
    d["eps"] = t.eps;
    return d;
}

LCNTrainingConfig TrainingFrom(float lr, float lr_min_frac, int lr_decay_epochs,
                               bool restore_best, float beta1, float beta2, float eps)
{
    LCNTrainingConfig t;
    t.lr = lr;
    t.lr_min_frac = lr_min_frac;
    t.lr_decay_epochs = lr_decay_epochs;
    t.restore_best = restore_best;
    t.beta1 = beta1;
    t.beta2 = beta2;
    t.eps = eps;
    return t;
}

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeWorldModel: a world model on a Boolean hypercube with "
              "frozen reservoir encoders and a trained predictor";
    m.attr("__version__") = HYPERCUBE_WORLDMODEL_VERSION;
    m.attr("cpp_version") = WorldModel::kVersion;

    m.def("paint_stripes", [](FloatArray src, size_t size) {
        const auto b = src.request();
        if (b.ndim != 1)
            throw std::invalid_argument("paint_stripes: src must be 1-D");
        py::array_t<float> out(static_cast<py::ssize_t>(size));
        PaintStripes(std::span<const float>(static_cast<const float*>(b.ptr),
                                            static_cast<size_t>(b.shape[0])),
                     std::span<float>(out.mutable_data(), size));
        return out;
    }, py::arg("src"), py::arg("size"),
       "Lay a short vector onto a field of the given size as contiguous stripes.");

    m.def("mean_abs", [](FloatArray x) {
        const auto b = x.request();
        return MeanAbs(std::span<const float>(static_cast<const float*>(b.ptr),
                                              static_cast<size_t>(b.size)));
    }, py::arg("x"), "Mean of absolute values.");

    m.def("rms", [](FloatArray x) {
        const auto b = x.request();
        return Rms(std::span<const float>(static_cast<const float*>(b.ptr),
                                          static_cast<size_t>(b.size)));
    }, py::arg("x"), "Root-mean-square.");

    // ── WorldModel ──

    py::class_<WorldModel>(m, "_WorldModel")
        .def(py::init([](size_t dim, size_t k,
                         uint64_t encoder_seed, uint64_t ic_seed,
                         float spectral_radius, float leak_rate, float input_scaling,
                         float output_scale,
                         size_t history_depth, size_t passes,
                         size_t z_max, size_t gather_span, bool tanh_last, uint64_t seed,
                         float lr, float lr_min_frac, int lr_decay_epochs, bool restore_best,
                         float beta1, float beta2, float eps) {
            WorldModelConfig cfg;
            cfg.encoder.dim = dim;
            cfg.encoder.seed = encoder_seed;
            cfg.encoder.ic_seed = ic_seed;
            cfg.encoder.spectral_radius = spectral_radius;
            cfg.encoder.leak_rate = leak_rate;
            cfg.encoder.input_scaling = input_scaling;
            cfg.encoder.output_scale = output_scale;
            cfg.encoder.history_depth = history_depth;
            cfg.encoder.passes = passes;
            cfg.k = k;
            cfg.predictor.z_max = z_max;
            cfg.predictor.gather_span = gather_span;
            cfg.predictor.tanh_last = tanh_last;
            cfg.predictor.seed = seed;
            cfg.predictor.training = TrainingFrom(lr, lr_min_frac, lr_decay_epochs,
                                                  restore_best, beta1, beta2, eps);
            return WorldModel::Create(cfg);
        }),
            py::arg("dim"), py::arg("k"),
            py::arg("encoder_seed"), py::arg("ic_seed"),
            py::arg("spectral_radius"), py::arg("leak_rate"), py::arg("input_scaling"),
            py::arg("output_scale"),
            py::arg("history_depth"), py::arg("passes"),
            py::arg("z_max"), py::arg("gather_span"), py::arg("tanh_last"), py::arg("seed"),
            py::arg("lr"), py::arg("lr_min_frac"), py::arg("lr_decay_epochs"),
            py::arg("restore_best"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"))

        .def_static("load", [](const std::filesystem::path& file) {
            return WorldModel::Load(file);
        }, py::arg("path"), "Read a model written by save().")

        .def("save", [](const WorldModel& self, const std::filesystem::path& file) {
            self.Save(file);
        }, py::arg("path"), "Write the config and the Predictor weights.")

        .def("config", [](const WorldModel& self) {
            const WorldModelConfig& c = self.Config();
            py::dict d;
            d["dim"] = c.encoder.dim;
            d["k"] = c.k;
            d["encoder_seed"] = c.encoder.seed;
            d["ic_seed"] = c.encoder.ic_seed;
            d["spectral_radius"] = c.encoder.spectral_radius;
            d["leak_rate"] = c.encoder.leak_rate;
            d["input_scaling"] = c.encoder.input_scaling;
            d["output_scale"] = self.ViewOutputScale();
            d["action_output_scale"] = self.ActionOutputScale();
            d["history_depth"] = c.encoder.history_depth;
            d["passes"] = self.RequestedPasses();   // as given, 0 included; not the resolved T
            d["z_max"] = c.predictor.z_max;          // resolved: 0 already replaced by k+1
            d["gather_span"] = c.predictor.gather_span;
            d["tanh_last"] = c.predictor.tanh_last;
            d["seed"] = c.predictor.seed;
            const py::dict t = TrainingDict(c.predictor.training);
            for (auto item : t)
                d[item.first] = item.second;
            return d;
        }, "Constructor knobs that rebuild this model. passes is as given "
           "(0 stays 0); z_max is resolved.")

        .def("view_output_scale", &WorldModel::ViewOutputScale)
        .def("action_output_scale", &WorldModel::ActionOutputScale)
        .def("set_view_output_scale", &WorldModel::SetViewOutputScale, py::arg("scale"))
        .def("set_action_output_scale", &WorldModel::SetActionOutputScale, py::arg("scale"))
        .def("suggest_view_output_scale", [](const WorldModel& self, FloatArray z, float target_rms) {
            const auto b = z.request();
            const auto* p = static_cast<const float*>(b.ptr);
            return self.SuggestViewOutputScale(
                std::span<const float>(p, static_cast<size_t>(b.size)), target_rms);
        }, py::arg("z"), py::arg("target_rms") = 1.f)
        .def("suggest_action_output_scale", [](const WorldModel& self, FloatArray za, float target_rms) {
            const auto b = za.request();
            const auto* p = static_cast<const float*>(b.ptr);
            return self.SuggestActionOutputScale(
                std::span<const float>(p, static_cast<size_t>(b.size)), target_rms);
        }, py::arg("za"), py::arg("target_rms") = 1.f)
        .def("fit_view_output_scale", [](WorldModel& self, FloatArray z, float target_rms) {
            const auto b = z.request();
            const auto* p = static_cast<const float*>(b.ptr);
            self.FitViewOutputScale(
                std::span<const float>(p, static_cast<size_t>(b.size)), target_rms);
        }, py::arg("z"), py::arg("target_rms") = 1.f)
        .def("fit_action_output_scale", [](WorldModel& self, FloatArray za, float target_rms) {
            const auto b = za.request();
            const auto* p = static_cast<const float*>(b.ptr);
            self.FitActionOutputScale(
                std::span<const float>(p, static_cast<size_t>(b.size)), target_rms);
        }, py::arg("za"), py::arg("target_rms") = 1.f)

        .def("encode", [](WorldModel& self, FloatArray fields) {
            const size_t n = self.FieldSize(), c = self.CodeSize();
            const Rows in = AsRows(fields, n, "fields");
            py::array_t<float> out = Matrix(in.count, c);
            float* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < in.count; ++i)
                    self.Encode(std::span<const float>(in.data + i * n, n),
                                std::span<float>(o + i * c, c));
            }
            return out;
        }, py::arg("fields"), "View codes, one row per field. Shape (rows, code_size).")

        .def("last_cube", [](const WorldModel& self) {
            return PointerToArray(self.LastCube(), self.FieldSize());
        }, "Scaled full view episode behind the most recent encode.")

        .def("last_raw_cube", [](const WorldModel& self) {
            return PointerToArray(self.LastRawCube(), self.FieldSize());
        }, "Unscaled full view episode behind the most recent encode.")

        .def("last_packed", [](const WorldModel& self) {
            return PointerToArray(self.LastPacked(), 2 * self.CodeSize());
        }, "Packed E(x) then E(a) from the most recent predict or accumulate.")

        .def("encode_action", [](WorldModel& self, FloatArray pictures) {
            const size_t c = self.CodeSize();
            const Rows in = AsRows(pictures, c, "pictures");
            py::array_t<float> out = Matrix(in.count, c);
            float* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < in.count; ++i)
                    self.EncodeAction(std::span<const float>(in.data + i * c, c),
                                      std::span<float>(o + i * c, c));
            }
            return out;
        }, py::arg("pictures"), "Action codes, one row per picture. Shape (rows, code_size).")

        .def("predict", [](WorldModel& self, FloatArray z, FloatArray za) {
            const size_t c = self.CodeSize();
            const Rows zi = AsRows(z, c, "z");
            const Rows ai = AsRows(za, c, "za");
            RequireSameRows(zi.count, ai.count, "predict");
            py::array_t<float> out = Matrix(zi.count, c);
            float* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < zi.count; ++i)
                {
                    const float* hat = self.Predict(std::span<const float>(zi.data + i * c, c),
                                                    std::span<const float>(ai.data + i * c, c));
                    std::memcpy(o + i * c, hat, c * sizeof(float));
                }
            }
            return out;
        }, py::arg("z"), py::arg("za"), "Predicted next view codes, one row per pair.")

        .def("rollout", [](WorldModel& self, FloatArray z0, FloatArray actions) {
            const size_t c = self.CodeSize();
            const Rows zi = AsRows(z0, c, "z0");
            const auto ab = actions.request();
            if (ab.ndim != 3)
                throw std::invalid_argument("rollout: actions must be 3-D (rows, H, code_size)");
            const size_t rows = static_cast<size_t>(ab.shape[0]);
            const size_t h = static_cast<size_t>(ab.shape[1]);
            if (static_cast<size_t>(ab.shape[2]) != c)
                throw std::invalid_argument("rollout: actions last dimension must be "
                                            + std::to_string(c));
            RequireSameRows(zi.count, rows, "rollout");
            const float* a = static_cast<const float*>(ab.ptr);
            py::array_t<float> out({static_cast<py::ssize_t>(rows),
                                    static_cast<py::ssize_t>(h + 1),
                                    static_cast<py::ssize_t>(c)});
            float* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < rows; ++i)
                    self.Rollout(std::span<const float>(zi.data + i * c, c),
                                 std::span<const float>(a + i * h * c, h * c),
                                 std::span<float>(o + i * (h + 1) * c, (h + 1) * c));
            }
            return out;
        }, py::arg("z0"), py::arg("actions"),
           "Chain predict over H action codes per row. Shape (rows, H + 1, code_size).")

        .def("pack", [](const WorldModel& self, FloatArray z, FloatArray za) {
            const size_t c = self.CodeSize();
            const Rows zi = AsRows(z, c, "z");
            const Rows ai = AsRows(za, c, "za");
            RequireSameRows(zi.count, ai.count, "pack");
            py::array_t<float> out = Matrix(zi.count, 2 * c);
            float* o = out.mutable_data();
            for (size_t i = 0; i < zi.count; ++i)
                self.Pack(std::span<const float>(zi.data + i * c, c),
                          std::span<const float>(ai.data + i * c, c),
                          std::span<float>(o + i * 2 * c, 2 * c));
            return out;
        }, py::arg("z"), py::arg("za"), "What the Predictor sees, one row per pair.")

        .def("begin_batch", &WorldModel::BeginBatch,
             "Clear the accumulated gradient. Call at the start of each batch.")

        .def("accumulate", [](WorldModel& self, FloatArray z, FloatArray za, FloatArray next) {
            const size_t c = self.CodeSize();
            const Rows zi = AsRows(z, c, "z");
            const Rows ai = AsRows(za, c, "za");
            const Rows ni = AsRows(next, c, "next");
            RequireSameRows(zi.count, ai.count, "accumulate");
            RequireSameRows(zi.count, ni.count, "accumulate");
            double sum = 0.0;
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < zi.count; ++i)
                    sum += self.Accumulate(std::span<const float>(zi.data + i * c, c),
                                           std::span<const float>(ai.data + i * c, c),
                                           std::span<const float>(ni.data + i * c, c));
            }
            return sum;
        }, py::arg("z"), py::arg("za"), py::arg("next"),
           "Forward, loss, backward for every row. Returns the summed loss, "
           "0.5 * SSE per pair.")

        .def("end_batch", [](WorldModel& self) {
            py::gil_scoped_release release;
            self.EndBatch();
        }, "One Adam step on the accumulated gradient.")

        .def("set_epoch", &WorldModel::SetEpoch, py::arg("epoch"), py::arg("num_epochs") = 0,
             "Apply the cosine schedule for this epoch.")
        .def("observe", [](WorldModel& self, float metric, int epoch) {
            py::gil_scoped_release release;
            self.Observe(metric, epoch);
        }, py::arg("metric"), py::arg("epoch"),
           "Snapshot the Predictor weights on a new low metric (restore_best only).")
        .def("restore_best", &WorldModel::RestoreBest,
             "Write the best-metric snapshot back (no-op if none).")
        .def("reset_training", &WorldModel::ResetTraining,
             "Forget Adam moments, step count, lr, and the best snapshot.")

        .def("weights", [](const WorldModel& self) { return VectorToArray(self.Weights()); },
             "Copy of the Predictor weights: depth, axis, tap, vertex.")
        .def("load_weights", [](WorldModel& self, FloatArray w) {
            const auto b = w.request();
            self.LoadWeights(std::span<const float>(static_cast<const float*>(b.ptr),
                                                    static_cast<size_t>(b.size)));
        }, py::arg("weights"), "Replace the Predictor weights; exact length required.")
        .def("grad", [](const WorldModel& self) { return VectorToArray(self.Grad()); },
             "Copy of the accumulated gradient, same layout as weights().")
        .def("add_grad", [](WorldModel& self, FloatArray g) {
            const auto b = g.request();
            self.AddGrad(std::span<const float>(static_cast<const float*>(b.ptr),
                                                static_cast<size_t>(b.size)));
        }, py::arg("grad"), "Sum a gradient of the same layout onto the accumulated one.")

        .def_property_readonly("field_size", &WorldModel::FieldSize)
        .def_property_readonly("code_size", &WorldModel::CodeSize)
        .def_property_readonly("k", &WorldModel::K)
        .def_property_readonly("dim", [](const WorldModel& self) { return self.Config().encoder.dim; })
        .def_property_readonly("passes", [](const WorldModel& self) { return self.Config().encoder.passes; })
        .def_property_readonly("action_passes", [](const WorldModel& self) { return self.ActionEncoderConfig().passes; })
        .def_property_readonly("num_weights", [](const WorldModel& self) { return self.Weights().size(); })
        .def_property_readonly("realized_spectral_radius", &WorldModel::RealizedSpectralRadius)
        .def_property_readonly("action_realized_spectral_radius", &WorldModel::ActionRealizedSpectralRadius);

    // ── Decoder ──

    py::class_<Decoder>(m, "_Decoder")
        .def(py::init([](size_t dim, size_t k, size_t z_max, size_t gather_span, bool tanh_last,
                         uint64_t seed, float lr, float lr_min_frac, int lr_decay_epochs,
                         bool restore_best, float beta1, float beta2, float eps) {
            DecoderConfig cfg;
            cfg.dim = dim;
            cfg.k = k;
            cfg.z_max = z_max;
            cfg.gather_span = gather_span;
            cfg.tanh_last = tanh_last;
            cfg.seed = seed;
            cfg.training = TrainingFrom(lr, lr_min_frac, lr_decay_epochs,
                                        restore_best, beta1, beta2, eps);
            return Decoder::Create(cfg);
        }),
            py::arg("dim"), py::arg("k"), py::arg("z_max"), py::arg("gather_span"),
            py::arg("tanh_last"), py::arg("seed"),
            py::arg("lr"), py::arg("lr_min_frac"), py::arg("lr_decay_epochs"),
            py::arg("restore_best"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"))

        .def_static("load", [](const std::filesystem::path& file) {
            return Decoder::Load(file);
        }, py::arg("path"), "Read a decoder written by save().")

        .def("save", [](const Decoder& self, const std::filesystem::path& file) {
            self.Save(file);
        }, py::arg("path"), "Write the config, the input scale, and the weights.")

        .def("config", [](const Decoder& self) {
            const DecoderConfig& c = self.Config();
            py::dict d;
            d["dim"] = c.dim;
            d["k"] = c.k;
            d["z_max"] = c.z_max;   // resolved
            d["gather_span"] = c.gather_span;
            d["tanh_last"] = c.tanh_last;
            d["seed"] = c.seed;
            const py::dict t = TrainingDict(c.training);
            for (auto item : t)
                d[item.first] = item.second;
            return d;
        }, "Constructor knobs that rebuild this decoder. z_max is resolved.")

        .def("decode", [](Decoder& self, FloatArray codes) {
            const size_t c = self.CodeSize(), n = self.FieldSize();
            const Rows in = AsRows(codes, c, "codes");
            py::array_t<float> out = Matrix(in.count, n);
            float* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < in.count; ++i)
                {
                    const float* f = self.Decode(std::span<const float>(in.data + i * c, c));
                    std::memcpy(o + i * n, f, n * sizeof(float));
                }
            }
            return out;
        }, py::arg("codes"), "Reconstructed fields, one row per code. Shape (rows, field_size).")

        .def("begin_batch", &Decoder::BeginBatch,
             "Clear the accumulated gradient. Call at the start of each batch.")
        .def("accumulate", [](Decoder& self, FloatArray codes, FloatArray targets) {
            const size_t c = self.CodeSize(), n = self.FieldSize();
            const Rows ci = AsRows(codes, c, "codes");
            const Rows ti = AsRows(targets, n, "targets");
            RequireSameRows(ci.count, ti.count, "accumulate");
            double sum = 0.0;
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < ci.count; ++i)
                    sum += self.Accumulate(std::span<const float>(ci.data + i * c, c),
                                           std::span<const float>(ti.data + i * n, n));
            }
            return sum;
        }, py::arg("codes"), py::arg("targets"),
           "Forward, loss, backward for every row. Returns the summed loss, "
           "0.5 * SSE over the field per pair.")
        .def("end_batch", [](Decoder& self) {
            py::gil_scoped_release release;
            self.EndBatch();
        }, "One Adam step on the accumulated gradient.")
        .def("set_epoch", &Decoder::SetEpoch, py::arg("epoch"), py::arg("num_epochs") = 0,
             "Apply the cosine schedule for this epoch.")
        .def("observe", [](Decoder& self, float metric, int epoch) {
            py::gil_scoped_release release;
            self.Observe(metric, epoch);
        }, py::arg("metric"), py::arg("epoch"),
           "Snapshot the weights on a new low metric (restore_best only).")
        .def("restore_best", &Decoder::RestoreBest,
             "Write the best-metric snapshot back (no-op if none).")
        .def("reset_training", &Decoder::ResetTraining,
             "Forget Adam moments, step count, lr, and the best snapshot.")

        .def("fit_input_scale", [](Decoder& self, FloatArray codes) {
            const auto b = codes.request();
            self.FitInputScale(std::span<const float>(static_cast<const float*>(b.ptr),
                                                      static_cast<size_t>(b.size)));
        }, py::arg("codes"),
           "Set the input scale to 1 / max |x| over every value of codes.")
        .def("set_input_scale", &Decoder::SetInputScale, py::arg("scale"))
        .def_property_readonly("input_scale", &Decoder::InputScale)

        .def("weights", [](const Decoder& self) { return VectorToArray(self.Weights()); },
             "Copy of the weights: depth, axis, tap, vertex.")
        .def("load_weights", [](Decoder& self, FloatArray w) {
            const auto b = w.request();
            self.LoadWeights(std::span<const float>(static_cast<const float*>(b.ptr),
                                                    static_cast<size_t>(b.size)));
        }, py::arg("weights"), "Replace the weights; exact length required.")
        .def("grad", [](const Decoder& self) { return VectorToArray(self.Grad()); },
             "Copy of the accumulated gradient, same layout as weights().")
        .def("add_grad", [](Decoder& self, FloatArray g) {
            const auto b = g.request();
            self.AddGrad(std::span<const float>(static_cast<const float*>(b.ptr),
                                                static_cast<size_t>(b.size)));
        }, py::arg("grad"), "Sum a gradient of the same layout onto the accumulated one.")

        .def_property_readonly("code_size", &Decoder::CodeSize)
        .def_property_readonly("field_size", &Decoder::FieldSize)
        .def_property_readonly("num_weights", [](const Decoder& self) { return self.Weights().size(); });
}
