#!/usr/bin/env python3
"""Validate non-owning zero-copy PyTorch ROCm Tensor descriptors."""

from __future__ import annotations

import argparse
import gc
import json
import weakref
from pathlib import Path


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def wrap(tensor, stream_device: str):
    from microllm import Tensor

    return Tensor.from_external(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(),
        tuple(tensor.shape), tuple(tensor.stride()),
        device=stream_device, owner=tensor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--run-id", default="pytorch-zero-copy")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if (arguments.device < 0 or arguments.rows <= 0 or arguments.width <= 0 or
            arguments.iterations <= 1):
        raise ValueError("device must be non-negative and workload dimensions positive")

    import torch
    from microllm import Stream, Tensor, add_out
    from microllm.profiling import profile_scope

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    profile = Path(arguments.profile)
    report_path = Path(arguments.report)
    for path in (profile, report_path):
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(f"refusing to replace {path}")
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("", encoding="utf-8")

    device = f"hip:{arguments.device}"
    torch_stream = torch.cuda.Stream(device=arguments.device)
    stream = Stream.from_external(int(torch_stream.cuda_stream), device=device)
    torch_left = torch.full((arguments.rows, arguments.width), 1.0,
                            device="cuda", dtype=torch.float32)
    torch_right = torch.full_like(torch_left, 2.0)
    torch_output = torch.empty_like(torch_left)
    left = wrap(torch_left, device)
    right = wrap(torch_right, device)
    output = wrap(torch_output, device)
    pointers_match = (
        left.data_ptr == torch_left.data_ptr() and
        right.data_ptr == torch_right.data_ptr() and
        output.data_ptr == torch_output.data_ptr())
    wrappers_non_owning = not left.owning and not right.owning and not output.owning

    with profile_scope("pytorch.zero_copy.add.first", output=profile,
                       run_id=arguments.run_id, emit_roctx=True):
        for _ in range(arguments.iterations):
            add_out(output, left, right, stream=stream)
        first_finish = torch.cuda.Event()
        first_finish.record(torch_stream)
        first_pending = not first_finish.query()
    first_finish.synchronize()
    first_error = float((torch_output - 3.0).abs().max().item())

    with profile_scope("pytorch.zero_copy.add.mutated", output=profile,
                       run_id=arguments.run_id, emit_roctx=True):
        with torch.cuda.stream(torch_stream):
            torch_left.fill_(10.0)
        for _ in range(arguments.iterations):
            add_out(output, left, right, stream=stream)
        second_finish = torch.cuda.Event()
        second_finish.record(torch_stream)
        second_pending = not second_finish.query()
    second_finish.synchronize()
    second_error = float((torch_output - 12.0).abs().max().item())

    transposed_owner = torch_left.transpose(0, 1)
    transposed = wrap(transposed_owner, device)
    noncontiguous_rejected = False
    try:
        add_out(output, transposed, right, stream=stream)
    except Exception:
        noncontiguous_rejected = True
    transposed.close()
    del transposed_owner
    short_storage_rejected = False
    try:
        Tensor.from_external(
            torch_left.data_ptr(), 4, tuple(torch_left.shape),
            tuple(torch_left.stride()), device=device, owner=torch_left)
    except Exception:
        short_storage_rejected = True

    left_reference = weakref.ref(torch_left)
    del torch_left
    gc.collect()
    owner_retained_by_wrapper = left_reference() is not None
    left.close()
    gc.collect()
    owner_released_after_close = left_reference() is None

    output_pointer = torch_output.data_ptr()
    output.close()
    right.close()
    wrapper_destroy_preserved_torch = torch_output.data_ptr() == output_pointer
    with torch.cuda.stream(torch_stream):
        torch_output.fill_(42.0)
    cleanup = torch.cuda.Event()
    cleanup.record(torch_stream)
    cleanup.synchronize()
    wrapper_destroy_preserved_torch = (
        wrapper_destroy_preserved_torch and
        float(torch_output.flatten()[0].item()) == 42.0)
    stream.close()

    report = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "rows": arguments.rows,
        "width": arguments.width,
        "iterations": arguments.iterations,
        "tensor_bytes": arguments.rows * arguments.width * 4,
        "wrapped_payload_bytes": arguments.rows * arguments.width * 4 * 3,
        "wrapper_copy_bytes": 0,
        "pointers_match": pointers_match,
        "wrappers_non_owning": wrappers_non_owning,
        "first_event_pending": first_pending,
        "second_event_pending": second_pending,
        "first_output_max_error": first_error,
        "mutated_input_output_max_error": second_error,
        "noncontiguous_rejected": noncontiguous_rejected,
        "short_storage_rejected": short_storage_rejected,
        "owner_retained_by_wrapper": owner_retained_by_wrapper,
        "owner_released_after_close": owner_released_after_close,
        "wrapper_destroy_preserved_torch": wrapper_destroy_preserved_torch,
    }
    required = (
        pointers_match and wrappers_non_owning and first_pending and second_pending and
        first_error == 0.0 and second_error == 0.0 and noncontiguous_rejected and
        short_storage_rejected and owner_retained_by_wrapper and
        owner_released_after_close and wrapper_destroy_preserved_torch)
    if not required:
        raise RuntimeError("PyTorch zero-copy Tensor gate failed")
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
