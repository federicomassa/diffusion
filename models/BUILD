load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    deps = [
        "@diffusion_pip_deps//tensorflow",
        "@diffusion_pip_deps//keras",
        "@diffusion_pip_deps//numpy",
    ],
)

py_library(
    name = "diffusion",
    srcs = ["diffusion.py"],
    deps = [
        ":models",
        "@diffusion_pip_deps//tensorflow",
        "@diffusion_pip_deps//numpy",
    ],
) 