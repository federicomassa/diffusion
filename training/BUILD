load("@rules_python//python:defs.bzl", "py_library", "py_binary")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "training",
    srcs = ["__init__.py"],
)

py_binary(
    name = "train_diffusion",
    srcs = ["train_diffusion.py"],
    deps = [
        ":training",
        "//data_utils:gaussian_data",
        "//models:diffusion",
        "@diffusion_pip_deps//tensorflow",
        "@diffusion_pip_deps//numpy",
        "@diffusion_pip_deps//matplotlib",
    ],
    python_version = "PY3",
) 