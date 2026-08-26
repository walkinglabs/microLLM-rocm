# microLLM-rocm documentation

This index separates user-facing framework documentation, developer documentation,
and chronological engineering evidence.

## Use the framework

- [Build from source](dev/build.md)
- [CMake Config beginner guide (中文)](dev/cmake-package.zh-CN.md)
- [Architecture](ARCHITECTURE.md)
- [Weight and safetensors API](WEIGHTS.md)
- [Hugging Face and verified Qwen2.5 workflow](HUGGINGFACE.md)
- [DeepSeek Distill support and flagship boundary](DEEPSEEK.md)
- [Operator contracts and PyTorch tolerances](OPERATOR_CONTRACTS.zh-CN.md)
- [Hardware and ROCm compatibility](COMPATIBILITY.md)
- [Benchmarks](../benchmarks/README.md)

## Develop the framework

- [Developer guide](dev/index.md)
- [Repository layout](dev/repository-layout.md)
- [Build system and toolchain](dev/build.md)
- [Testing and evidence](dev/testing.md)
- [Adding or optimizing an operator](dev/operator-development.md)
- [Profiling](dev/profiling.md)
- [microLLM/PyTorch alignment](dev/alignment.md)
- [Data-parallel training](dev/distributed-training.md)
- [Contribution process](../CONTRIBUTING.md)
- [Task contract](TASK_CONTRACT.md)

## Project governance and evidence

- [Project charter](PROJECT_CHARTER.md)
- [Current evidence status](development/STATUS.md)
- [Roadmap](development/ROADMAP.md)
- [Explicit next steps](development/NEXT_STEPS.md)
- [Chronological development records](development/README.md)
- [0→1 performance optimization log](optimization-log/README.md)
- [Living optimization blog](optimization-log/BLOG.zh-CN.md)

## Course

The beginner course, N0–N10, and PA0–PA2 are maintained on the separate
[`tutorial/beginner-course`](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course)
branch. It is course-only and runs examples/tests from an independent `main` checkout.
Framework releases and course publication have independent acceptance gates.
