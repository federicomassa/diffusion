load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "check_gpu",
    srcs = ["check_gpu.py"],
    deps = [
        "@diffusion_pip_deps//tensorflow",
    ],
)

py_binary(
    name = "test_keras_install",
    srcs = ["test_keras_install.py"],
    deps = [
        "@diffusion_pip_deps//tensorflow",
        "@diffusion_pip_deps//keras",
        "@diffusion_pip_deps//numpy",
        "@diffusion_pip_deps//matplotlib",
    ],
) 