load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "run_diffusion",
    srcs = ["run_diffusion.py"],
    deps = [
        "//training:train_diffusion",
    ],
    python_version = "PY3",
    data = [
        "//data_utils:gaussian_data",
        "//models:diffusion",
    ],
    args = [
        "--results-dir=/home/federico/results",
        "--n-samples=5000",
        "--epochs=30",
    ],
)

# Make shell scripts available
sh_binary(
    name = "run_diffusion_training",
    srcs = ["run_diffusion_training.sh"],
    data = [
        "//training:train_diffusion",
    ],
) 