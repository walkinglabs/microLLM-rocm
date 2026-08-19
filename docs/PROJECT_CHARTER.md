# Project charter

## Purpose

microLLM-rocm demonstrates that a compact, inspectable C++/HIP engine can train
and run real small language models on AMD GPUs. It serves engine developers,
ROCm learners, and contributors who need reproducible operator and system-level
evidence.

## Required outcomes

- independent C++/HIP training and inference engine;
- CPU references for numerical validation;
- readable and optimized HIP implementations of core LLM operators;
- Model-S and Model-M pretraining, checkpoint resume, SFT, and generation;
- optional stable C, Python, and PyTorch Custom Op integration layers;
- operator and end-to-end comparisons against equivalent PyTorch ROCm paths;
- single-, dual-, and four-GPU correctness and scaling reports when hardware is
  available;
- compatibility reports for explicitly tested AMD Instinct and Radeon systems;
- a course that consumes the same engine rather than a separate exercise engine.

## Performance claim policy

There is no universal “faster than PyTorch” claim. Every comparison is scoped by
model, operator, shape, stride, dtype, batch, context, GPU architecture, software
versions, warm-up, repetitions, and numerical tolerance. Results below the target
remain in the report.

The target is to meet or exceed PyTorch ROCm on selected, published benchmark
matrix entries while explaining gaps elsewhere.

## Non-goals for the first public release

- every model family and arbitrary graph compilation;
- cross-node training, FSDP, pipeline parallelism, or every quantization format;
- silently supporting untested Radeon/Instinct combinations;
- removing readable reference paths after adding an optimization.

## Model names

Model-S is approximately 15.6M parameters. Model-M targets approximately 31M
parameters. Weight footprint is always reported with dtype; optimizer and
activation memory are reported separately.
