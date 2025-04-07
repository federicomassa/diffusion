load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    deps = [
        "@pip_deps//:tensorflow",
        "@pip_deps//:keras",
        "@pip_deps//:numpy",
    ],
) 