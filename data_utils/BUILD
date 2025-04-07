load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "data_utils",
    srcs = ["__init__.py"],
)

py_library(
    name = "gaussian_data",
    srcs = ["gaussian_data.py"],
    deps = [
        ":data_utils",
        "@diffusion_pip_deps//numpy",
        "@diffusion_pip_deps//matplotlib",
    ],
) 