#!/usr/bin/env python3
"""Python Transformers/PyTorch baseline for official HF model matrix rows."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

try:
    from .hf_internal_parameter_mapping import internal_parameter
except ImportError:
    from hf_internal_parameter_mapping import internal_parameter


MODES = {"infer", "train"}
COMMON_FILES = ("config", "weights")


def comma_list(text: str) -> list[str]:
    values = text.split(",")
    if not values or any(value not in MODES for value in values):
        raise argparse.ArgumentTypeError("modes must be infer, train, or infer,train")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("modes cannot contain duplicates")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "bf16_amp"), default="fp32")
    parser.add_argument("--modes", default="infer,train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-up-parameters-output", type=Path)
    parser.add_argument("--gate-up-gradients-output", type=Path)
    parser.add_argument("--all-parameters-output", type=Path)
    parser.add_argument("--all-gradients-output", type=Path)
    parser.add_argument("--all-moments-output", type=Path)
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("infer", "train"), help=argparse.SUPPRESS)
    result = parser.parse_args()
    result.modes = comma_list(result.modes)
    if not result.manifest.is_file():
        parser.error(f"manifest does not exist: {result.manifest}")
    return result


def load_manifest(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise RuntimeError("HF matrix manifest must have schema_version 1 and models list")
    names: set[str] = set()
    for model in document["models"]:
        required = {
            "name", "revision", "parameter_count", "loaded_tensors", "config", "weights",
            "inference", "training",
        }
        missing = required - model.keys()
        if missing:
            raise RuntimeError(f"HF model entry is missing fields: {sorted(missing)}")
        if model["name"] in names:
            raise RuntimeError(f"duplicate HF model name: {model['name']}")
        names.add(model["name"])
        token_ids = model["inference"].get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"{model['name']} inference.token_ids must be non-empty")
    return document["models"]


def prepare_device(name: str, allow_fallback: bool) -> tuple[torch.device, str]:
    workaround = "none"
    if name == "cuda" and torch.version.hip:
        public_count = torch.cuda.device_count()
        runtime_count = torch._C._cuda_getDeviceCount()
        if public_count == 0 and runtime_count > 0:
            if not allow_fallback:
                raise RuntimeError(
                    "AMDSMI reports zero devices while HIP runtime sees devices; "
                    "pass --allow-amdsmi-fallback to use the explicit runtime fallback"
                )
            torch.cuda._device_count_amdsmi = lambda: -1
            workaround = "amdsmi_zero_fallback_to_hip_runtime"
    if name == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() == 0):
        raise RuntimeError("PyTorch CUDA/ROCm device requested but unavailable")
    return torch.device(name), workaround


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def memory_fields(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "device_current_allocated_bytes": 0,
            "device_peak_allocated_bytes": 0,
            "device_current_reserved_bytes": 0,
            "device_peak_reserved_bytes": 0,
        }
    return {
        "device_current_allocated_bytes": torch.cuda.memory_allocated(device),
        "device_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device_current_reserved_bytes": torch.cuda.memory_reserved(device),
        "device_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def load_model(model: dict, device: torch.device, dtype_name: str):
    clear_device(device)
    start = time.perf_counter()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    loaded = AutoModelForCausalLM.from_pretrained(
        Path(model["config"]).parent,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    synchronize(device)
    finish = time.perf_counter()
    parameter_count = sum(parameter.numel() for parameter in loaded.parameters())
    tensor_count = len(loaded.state_dict())
    if parameter_count != model["parameter_count"]:
        raise RuntimeError(f"{model['name']} PyTorch parameter count changed")
    if tensor_count not in {model["loaded_tensors"], model["loaded_tensors"] + 1}:
        raise RuntimeError(
            f"{model['name']} PyTorch state Tensor count {tensor_count} "
            f"is incompatible with checkpoint count {model['loaded_tensors']}"
        )
    return loaded, (finish - start) * 1000.0, parameter_count, tensor_count


def common(model: dict, loaded, device: torch.device, workaround: str,
           load_ms: float, parameter_count: int, tensor_count: int,
           dtype_name: str) -> dict:
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {
        "schema_version": 1,
        "record_type": "single_gpu_pytorch_hf_model_measurement",
        "status": "pass",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "device_name": properties.name if properties else "host CPU",
        "architecture": properties.gcnArchName if properties else "host",
        "device_discovery_workaround": workaround,
        "compute_dtype": dtype_name,
        "model": model["name"],
        "revision": model["revision"],
        "parameter_count": parameter_count,
        "fp32_weight_bytes": parameter_count * 4,
        "resident_weight_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in loaded.parameters()
        ),
        "loaded_tensors": model["loaded_tensors"],
        "pytorch_state_tensors": tensor_count,
        "load_ms": load_ms,
    }


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "bf16_amp":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def infer(model: dict, loaded, device: torch.device, base: dict) -> dict:
    inference = model["inference"]
    input_ids = torch.tensor([inference["token_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    warmup = int(inference.get("warmup", 0))
    steps = int(inference.get("steps", 1))
    prefill_warmup = int(inference.get("prefill_warmup", warmup))
    prefill_steps = int(inference.get("prefill_steps", steps))

    def generate_once():
        return loaded.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=inference["new_tokens"],
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.config.eos_token_id,
        )[0, input_ids.shape[1]:].tolist()

    with torch.inference_mode(), autocast_context(device, base["compute_dtype"]):
        for _ in range(prefill_warmup):
            loaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(prefill_steps):
            loaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        synchronize(device)
        finish = time.perf_counter()
        warmup_start = time.perf_counter()
        for _ in range(warmup):
            generate_once()
        synchronize(device)
        warmup_finish = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        generation_start = time.perf_counter()
        suffix = []
        for _ in range(steps):
            current = generate_once()
            if suffix and current != suffix:
                raise RuntimeError(f"{model['name']} generation changed across steps")
            suffix = current
        synchronize(device)
        generation_finish = time.perf_counter()
    expected = inference.get("expected_generated_tokens")
    if expected is not None and suffix != expected:
        raise RuntimeError(f"{model['name']} PyTorch generated tokens changed: {suffix}")
    forward_ms = (finish - start) * 1000.0
    generation_ms = (generation_finish - generation_start) * 1000.0
    warmup_ms = (warmup_finish - warmup_start) * 1000.0
    measured_tokens = len(suffix) * steps
    base.update({
        "mode": "infer",
        "measurement_profile": "comparison" if warmup > 0 or steps > 1 else "smoke",
        "token_count": input_ids.shape[1],
        "warmup": warmup,
        "steps": steps,
        "prefill_warmup": prefill_warmup,
        "prefill_steps": prefill_steps,
        "warmup_ms": warmup_ms,
        "measured_tokens": measured_tokens,
        "generated_tokens": suffix,
        "forward_ms": forward_ms / prefill_steps,
        "prefill_tokens_per_second": (
            input_ids.shape[1] * prefill_steps * 1000.0 / forward_ms
        ),
        "generation_ms": generation_ms,
        "mean_generation_ms": generation_ms / steps,
        "decode_tokens_per_second": measured_tokens * 1000.0 / generation_ms,
        "decode_milliseconds_per_token": generation_ms / measured_tokens,
        **memory_fields(device),
    })
    return base


def internal_state(loaded, gradients: bool,
                   gate_up_only: bool = False) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in loaded.named_parameters():
        mapped = internal_parameter(name)
        if mapped is None:
            raise RuntimeError(f"unmapped PyTorch parameter: {name}")
        internal, transpose = mapped
        if gate_up_only and not (
                internal.endswith(".gate_proj.weight") or
                internal.endswith(".up_proj.weight")):
            continue
        source = parameter.grad if gradients else parameter
        if source is None:
            raise RuntimeError(f"missing parameter gradient: {name}")
        tensor = source.detach().float()
        if transpose:
            tensor = tensor.T
        result[internal] = tensor.contiguous().cpu()
    if not result:
        raise RuntimeError("PyTorch Tensor selection is empty")
    return result


def save_gate_up(path: Path, loaded, gradients: bool) -> tuple[int, int]:
    from safetensors.torch import save_file
    state = internal_state(loaded, gradients, True)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, path)
    return len(state), sum(tensor.numel() for tensor in state.values())


def save_all(path: Path, loaded, gradients: bool) -> tuple[int, int]:
    from safetensors.torch import save_file
    state = internal_state(loaded, gradients, False)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, path)
    return len(state), sum(tensor.numel() for tensor in state.values())


def save_all_moments(path: Path, loaded,
                     optimizer: torch.optim.Optimizer) -> tuple[int, int, int]:
    from safetensors.torch import save_file
    result = {}
    steps = set()
    for name, parameter in loaded.named_parameters():
        mapped = internal_parameter(name)
        if mapped is None:
            raise RuntimeError(f"unmapped PyTorch parameter: {name}")
        internal, transpose = mapped
        state = optimizer.state.get(parameter, {})
        if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
            raise RuntimeError(f"missing AdamW state: {name}")
        step_value = state["step"]
        steps.add(int(step_value.item() if torch.is_tensor(step_value) else step_value))
        for suffix, field in (("first_moment", "exp_avg"),
                              ("second_moment", "exp_avg_sq")):
            tensor = state[field].detach().float()
            if transpose:
                tensor = tensor.T
            result[f"{internal}.adamw.{suffix}"] = tensor.contiguous().cpu()
    if len(steps) != 1 or not result:
        raise RuntimeError("AdamW state has inconsistent steps or no Tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, path)
    return (len(result), sum(tensor.numel() for tensor in result.values()),
            next(iter(steps)))


def train(model: dict, loaded, device: torch.device, base: dict,
          parameter_output: Path | None = None,
          gradient_output: Path | None = None,
          all_parameter_output: Path | None = None,
          all_gradient_output: Path | None = None,
          all_moment_output: Path | None = None) -> dict:
    training = model["training"]
    all_tokens = [int(value) for value in training["tokens"].split(",")]
    batch = int(training.get("batch", 1))
    if batch <= 0:
        raise RuntimeError(f"{model['name']} training batch must be positive")
    inputs = torch.tensor([all_tokens[:-1]], dtype=torch.long, device=device).repeat(batch, 1)
    targets = torch.tensor([all_tokens[1:]], dtype=torch.long, device=device).repeat(batch, 1)
    optimizer = torch.optim.AdamW(
        loaded.parameters(), lr=float(training["learning_rate"]),
        betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.01
    )
    observed = dict(loaded.named_parameters()).get("model.norm.weight")
    if observed is None:
        raise RuntimeError(f"{model['name']} is missing model.norm.weight")
    warmup = int(training.get("warmup", 0))
    steps = int(training.get("steps", 1))
    if (gradient_output is not None or all_gradient_output is not None) and \
            (warmup != 0 or steps != 1):
        raise RuntimeError(
            "gradient output requires warmup 0 and steps 1")
    gradient_count = 0
    gradient_elements = 0
    parameter_count_output = 0
    parameter_elements = 0
    all_gradient_count = 0
    all_gradient_elements = 0
    all_parameter_count = 0
    all_parameter_elements = 0
    all_moment_count = 0
    all_moment_elements = 0
    all_moment_step = 0

    def train_once(capture: bool = False):
        nonlocal gradient_count, gradient_elements
        nonlocal all_gradient_count, all_gradient_elements
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, base["compute_dtype"]):
            logits = loaded(input_ids=inputs, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        synchronize(device)
        if capture and gradient_output is not None:
            gradient_count, gradient_elements = save_gate_up(
                gradient_output, loaded, True)
        if capture and all_gradient_output is not None:
            all_gradient_count, all_gradient_elements = save_all(
                all_gradient_output, loaded, True)
        optimizer_start = time.perf_counter()
        optimizer.step()
        synchronize(device)
        return float(loss.detach()), (time.perf_counter() - optimizer_start) * 1000.0

    warmup_start = time.perf_counter()
    for _ in range(warmup):
        train_once()
    synchronize(device)
    warmup_finish = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = float(observed[0].detach())
    first_loss = 0.0
    final_loss = 0.0
    optimizer_ms = 0.0
    start = time.perf_counter()
    for iteration in range(steps):
        loss_value, current_optimizer_ms = train_once(capture=iteration == steps - 1)
        if iteration == 0:
            first_loss = loss_value
        final_loss = loss_value
        optimizer_ms += current_optimizer_ms
    synchronize(device)
    finish = time.perf_counter()
    if parameter_output is not None:
        parameter_count_output, parameter_elements = save_gate_up(
            parameter_output, loaded, False)
    if all_parameter_output is not None:
        all_parameter_count, all_parameter_elements = save_all(
            all_parameter_output, loaded, False)
    if all_moment_output is not None:
        all_moment_count, all_moment_elements, all_moment_step = \
            save_all_moments(all_moment_output, loaded, optimizer)
    after = float(observed[0].detach())
    step_ms = (finish - start) * 1000.0
    trained_tokens = inputs.numel() * steps
    if not math.isfinite(final_loss) or before == after:
        raise RuntimeError(f"{model['name']} PyTorch train step did not update")
    base.update({
        "mode": "train",
        "measurement_profile": (
            "diagnostic" if gradient_output is not None or
                            all_gradient_output is not None or
                            all_moment_output is not None else
            "comparison" if warmup > 0 or steps > 1 else "smoke"),
        "gate_up_parameters_output_written": parameter_output is not None,
        "gate_up_parameter_tensors": parameter_count_output,
        "gate_up_parameter_elements": parameter_elements,
        "gate_up_gradients_output_written": gradient_output is not None,
        "gate_up_gradient_tensors": gradient_count,
        "gate_up_gradient_elements": gradient_elements,
        "all_parameters_output_written": all_parameter_output is not None,
        "all_parameter_tensors": all_parameter_count,
        "all_parameter_elements": all_parameter_elements,
        "all_gradients_output_written": all_gradient_output is not None,
        "all_gradient_tensors": all_gradient_count,
        "all_gradient_elements": all_gradient_elements,
        "all_moments_output_written": all_moment_output is not None,
        "all_moment_tensors": all_moment_count,
        "all_moment_elements": all_moment_elements,
        "all_moment_step": all_moment_step,
        "warmup": warmup,
        "steps": steps,
        "batch": batch,
        "context": inputs.shape[1],
        "warmup_ms": (warmup_finish - warmup_start) * 1000.0,
        "trained_tokens": trained_tokens,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss": final_loss,
        "observed_parameter_before": before,
        "observed_parameter_after": after,
        "parameter_changed": True,
        "step_ms": step_ms,
        "measured_ms": step_ms,
        "mean_step_ms": step_ms / steps,
        "optimizer_ms": optimizer_ms,
        "mean_optimizer_ms": optimizer_ms / steps,
        "tokens_per_second": trained_tokens * 1000.0 / step_ms,
        "milliseconds_per_token": step_ms / trained_tokens,
        **memory_fields(device),
    })
    return base


def run_worker(model: dict, mode: str, device_name: str, allow_fallback: bool,
               dtype_name: str, parameter_output: Path | None = None,
               gradient_output: Path | None = None,
               all_parameter_output: Path | None = None,
               all_gradient_output: Path | None = None,
               all_moment_output: Path | None = None) -> dict:
    device, workaround = prepare_device(device_name, allow_fallback)
    loaded, load_ms, parameter_count, tensor_count = load_model(model, device, dtype_name)
    base = common(model, loaded, device, workaround, load_ms, parameter_count, tensor_count,
                  dtype_name)
    return infer(model, loaded, device, base) if mode == "infer" else train(
        model, loaded, device, base, parameter_output, gradient_output,
        all_parameter_output, all_gradient_output, all_moment_output
    )


def unavailable(model: dict, mode: str, fields: list[str]) -> dict:
    return {
        "schema_version": 1,
        "record_type": "single_gpu_pytorch_hf_model_measurement",
        "framework": "pytorch",
        "model": model["name"],
        "revision": model["revision"],
        "mode": mode,
        "status": "unavailable",
        "missing_inputs": fields,
    }


def main() -> int:
    args = options()
    models = load_manifest(args.manifest)
    by_name = {model["name"]: model for model in models}
    if (args.worker_model is None) != (args.worker_mode is None):
        raise RuntimeError("worker model and mode must be provided together")
    if (args.gate_up_parameters_output is not None or
            args.gate_up_gradients_output is not None or
            args.all_parameters_output is not None or
            args.all_gradients_output is not None or
            args.all_moments_output is not None) and \
            args.worker_mode != "train":
        raise RuntimeError("Tensor outputs require train worker mode")
    if ((args.gate_up_parameters_output is not None or
         args.gate_up_gradients_output is not None) and
            (args.all_parameters_output is not None or
             args.all_gradients_output is not None)):
        raise RuntimeError(
            "gate/up and all-Tensor outputs are mutually exclusive")
    if args.worker_model is not None:
        if args.worker_model not in by_name:
            raise RuntimeError(f"unknown worker model: {args.worker_model}")
        print(json.dumps(run_worker(by_name[args.worker_model], args.worker_mode, args.device,
                                    args.allow_amdsmi_fallback, args.dtype,
                                    args.gate_up_parameters_output,
                                    args.gate_up_gradients_output,
                                    args.all_parameters_output,
                                    args.all_gradients_output,
                                    args.all_moments_output), sort_keys=True))
        return 0

    records: list[dict] = []
    unavailable_count = 0
    for model in models:
        missing = [field for field in COMMON_FILES if not Path(model[field]).is_file()]
        for mode in args.modes:
            if missing:
                records.append(unavailable(model, mode, missing))
                unavailable_count += 1
                continue
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--manifest", str(args.manifest), "--device", args.device,
                "--dtype", args.dtype,
                "--worker-model", model["name"], "--worker-mode", mode,
            ]
            if args.allow_amdsmi_fallback:
                command.append("--allow-amdsmi-fallback")
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            records.append(json.loads(completed.stdout))
    status = "incomplete" if unavailable_count else "pass"
    summary = {
        "schema_version": 1,
        "record_type": "single_gpu_pytorch_hf_model_matrix_summary",
        "status": status,
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "transformers_version": transformers.__version__,
        "device": args.device,
        "compute_dtype": args.dtype,
        "model_count": len(models),
        "requested_modes": args.modes,
        "measurement_count": len(records) - unavailable_count,
        "unavailable_count": unavailable_count,
    }
    lines = [*(json.dumps(record, sort_keys=True) for record in records),
             json.dumps(summary, sort_keys=True)]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if not unavailable_count or args.allow_unavailable else 2


if __name__ == "__main__":
    raise SystemExit(main())
