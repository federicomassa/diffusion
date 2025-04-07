# Diffusion

A Python project for diffusion models.

## Building with Bazel

This project uses Bazel as the build system. 

### Prerequisites

- Install [Bazel](https://bazel.build/install)
- Python 3.12

### Building and Running

Build all targets:
```bash
bazel build //...
```

Run a specific binary:
```bash
bazel run //:check_gpu
bazel run //:test_keras_install
```

### Development

To add a new dependency, add it to `third_party/requirements.txt` and run:
```bash
bazel sync --only=pip_deps
```