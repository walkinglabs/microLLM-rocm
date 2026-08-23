#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TorchTraceSession, load_jsonl, write_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PyTorch side of a microLLM alignment experiment")
    parser.add_argument("--input", required=True, type=Path, help="directory produced by microllm_alignment")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-id", default="tiny-alignment")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-captured-elements", type=int, default=4096)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions <= 0 or args.max_captured_elements <= 0:
        parser.error("warmup must be non-negative; repetitions and capture limit must be positive")
    return args


def load_parameters(path, device):
    parameters = {}
    for record in load_jsonl(path):
        if record["kind"] != "parameter" or record["values_truncated"]:
            raise ValueError(f"parameter trace is incomplete: {record['name']}")
        tensor = torch.tensor(record["values"], dtype=torch.float32, device=device)
        parameters[record["name"]] = tensor.reshape(record["shape"])
    if not parameters:
        raise ValueError("parameter trace is empty")
    return parameters


def differentiable_parameters(parameters):
    return {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameters.items()
    }


def rope(value, sequence_dim=2, position_offset=0, base=10000.0):
    width = value.shape[-1]
    positions = torch.arange(
        position_offset,
        position_offset + value.shape[sequence_dim],
        dtype=value.dtype,
        device=value.device,
    )
    frequencies = base ** (-torch.arange(0, width, 2, dtype=value.dtype,
                                          device=value.device) / width)
    angles = positions[:, None] * frequencies[None, :]
    view_shape = [1] * value.dim()
    view_shape[sequence_dim] = value.shape[sequence_dim]
    view_shape[-1] = width // 2
    cosine = torch.cos(angles).reshape(view_shape)
    sine = torch.sin(angles).reshape(view_shape)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


def causal_softmax(scores):
    sequence = scores.shape[-1]
    future = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool,
                                   device=scores.device), diagonal=1)
    return torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)


def causal_gqa_attention(query, key, value, repeats, scale):
    if repeats != 1:
        key = torch.repeat_interleave(key, repeats, dim=1)
        value = torch.repeat_interleave(value, repeats, dim=1)
    return causal_softmax(torch.matmul(query, key.transpose(-2, -1)) * scale) @ value


def causal_gqa_attention_bthd(query, key, value, repeats, scale):
    if repeats != 1:
        key = torch.repeat_interleave(key, repeats, dim=1)
        value = torch.repeat_interleave(value, repeats, dim=2)
    probabilities = causal_softmax(
        torch.matmul(query, key.transpose(-2, -1)) * scale)
    return torch.matmul(probabilities, value.transpose(1, 2)).transpose(1, 2)


def rms_norm(value, weight, epsilon=1.0e-5):
    return F.rms_norm(value, (value.shape[-1],), weight, epsilon)


def model_forward(session, parameters, tokens, config):
    op = lambda name, fn: session.measure("operator", name, fn)

    def body():
        session.record("input", "model.tokens", tokens)
        hidden = session.measure(
            "layer", "model.embedding",
            lambda: op("embedding", lambda: F.embedding(tokens, parameters["token_embedding.weight"])),
        )

        for layer in range(config["layers"]):
            prefix = f"blocks.{layer}"

            def block_forward(hidden_state=hidden, prefix=prefix):
                normalized = op(
                    "rms_norm",
                    lambda: rms_norm(hidden_state, parameters[f"{prefix}.attention_norm.weight"]),
                )
                flat = op("reshape", lambda: normalized.reshape(-1, config["dimension"]))
                query = op("matmul", lambda: flat @ parameters[f"{prefix}.attention.q_proj.weight"])
                query = op("reshape", lambda: query.reshape(
                    tokens.shape[0], tokens.shape[1], config["heads"],
                    config["dimension"] // config["heads"]))
                key = op("matmul", lambda: flat @ parameters[f"{prefix}.attention.k_proj.weight"])
                key = op("reshape", lambda: key.reshape(
                    tokens.shape[0], tokens.shape[1], config["kv_heads"],
                    config["dimension"] // config["heads"]))
                value = op("matmul", lambda: flat @ parameters[f"{prefix}.attention.v_proj.weight"])
                value = op("reshape", lambda: value.reshape(
                    tokens.shape[0], tokens.shape[1], config["kv_heads"],
                    config["dimension"] // config["heads"]))
                query = op("transpose", lambda: query.transpose(1, 2))
                query = op("rope", lambda: rope(query, base=config["rope_base"]))
                key = op("transpose", lambda: key.transpose(1, 2))
                key = op("rope", lambda: rope(key, base=config["rope_base"]))
                repeats = config["heads"] // config["kv_heads"]
                context = op(
                    "causal_gqa_attention_bthd",
                    lambda: causal_gqa_attention_bthd(
                        query, key, value, repeats,
                        1.0 / math.sqrt(config["dimension"] // config["heads"])),
                )
                context = op("reshape", lambda: context.reshape(-1, config["dimension"]))
                attention = op("matmul", lambda: context @ parameters[f"{prefix}.attention.o_proj.weight"])
                attention = op("reshape", lambda: attention.reshape(
                    tokens.shape[0], tokens.shape[1], config["dimension"]))
                residual = op("add", lambda: hidden_state + attention)

                normalized_ffn = op(
                    "rms_norm",
                    lambda: rms_norm(residual, parameters[f"{prefix}.ffn_norm.weight"]),
                )
                flat_ffn = op("reshape", lambda: normalized_ffn.reshape(-1, config["dimension"]))
                gate = op("matmul", lambda: flat_ffn @ parameters[f"{prefix}.feed_forward.gate_proj.weight"])
                up = op("matmul", lambda: flat_ffn @ parameters[f"{prefix}.feed_forward.up_proj.weight"])
                activated = op("swiglu", lambda: F.silu(gate) * up)
                down = op("matmul", lambda: activated @ parameters[f"{prefix}.feed_forward.down_proj.weight"])
                down = op("reshape", lambda: down.reshape(
                    tokens.shape[0], tokens.shape[1], config["dimension"]))
                return op("add", lambda: residual + down)

            hidden = session.measure("layer", f"model.blocks.{layer}", block_forward)

        hidden = session.measure(
            "layer", "model.final_norm",
            lambda: op("rms_norm", lambda: rms_norm(hidden, parameters["final_norm.weight"])),
        )
        flat = op("reshape", lambda: hidden.reshape(-1, config["dimension"]))
        logits = op("matmul", lambda: flat @ parameters["output_head.weight"])
        logits = op("reshape", lambda: logits.reshape(
            tokens.shape[0], tokens.shape[1], config["vocabulary_size"]))
        session.record("output", "model.logits", logits)
        return logits

    return session.measure("model", "model.forward", body)


def model_loss(session, parameters, tokens, targets, config):
    logits = model_forward(session, parameters, tokens, config)
    return F.cross_entropy(
        logits.reshape(-1, config["vocabulary_size"]), targets.reshape(-1)
    )


def make_session(args, phase, device, **overrides):
    options = dict(
        record_operators=True,
        record_layers=True,
        record_model=True,
        capture_values=True,
        max_captured_elements=args.max_captured_elements,
    )
    options.update(overrides)
    return TorchTraceSession("pytorch", args.run_id, phase, device, **options)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA/ROCm device requested but unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters(args.input / "microllm_parameters.jsonl", device)
    config = json.loads((args.input / "microllm_run.json").read_text())["model"]
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    targets = torch.tensor([[1, 2, 3, 0]], dtype=torch.int64, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model_forward(make_session(args, "disabled", device,
                                       record_operators=False, record_layers=False,
                                       record_model=False, capture_values=False),
                          parameters, tokens, config)

        values = make_session(args, "values", device)
        values.iteration = 0
        model_forward(values, parameters, tokens, config)
        write_jsonl(args.output / "pytorch_values.jsonl", values.records)

        operators = make_session(
            args, "operator_timing", device,
            record_operators=True, record_layers=False, record_model=False,
            capture_values=False,
        )
        for iteration in range(args.repetitions):
            operators.iteration = iteration
            model_forward(operators, parameters, tokens, config)
        write_jsonl(args.output / "pytorch_operator_timing.jsonl", operators.records)

        layers = make_session(
            args, "layer_timing", device,
            record_operators=False, record_layers=True, record_model=True,
            capture_values=False,
        )
        for iteration in range(args.repetitions):
            layers.iteration = iteration
            model_forward(layers, parameters, tokens, config)
        write_jsonl(args.output / "pytorch_layer_timing.jsonl", layers.records)

    training_parameters = differentiable_parameters(parameters)
    disabled = make_session(
        args, "disabled", device,
        record_operators=False, record_layers=False, record_model=False,
        capture_values=False,
    )
    training_loss = model_loss(
        disabled, training_parameters, tokens, targets, config
    )
    training_loss.backward()
    training_values = make_session(
        args, "training_values", device,
        record_operators=False, record_layers=False, record_model=False,
        capture_values=True,
    )
    training_values.record("output", "training.loss", training_loss)
    for name, parameter in training_parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for parameter: {name}")
        training_values.record("parameter", f"gradient.{name}", parameter.grad)
    write_jsonl(args.output / "pytorch_training_values.jsonl", training_values.records)

    for _ in range(args.warmup):
        warmup_parameters = differentiable_parameters(parameters)
        model_loss(disabled, warmup_parameters, tokens, targets, config).backward()
    backward = make_session(
        args, "backward_timing", device,
        record_operators=False, record_layers=False, record_model=True,
        capture_values=False,
    )
    for iteration in range(args.repetitions):
        timing_parameters = differentiable_parameters(parameters)
        loss = model_loss(disabled, timing_parameters, tokens, targets, config)
        backward.iteration = iteration

        def run_backward(loss=loss):
            loss.backward()
            return loss

        backward.measure("model", "model.backward", run_backward)
    write_jsonl(args.output / "pytorch_backward_timing.jsonl", backward.records)

    metadata = {
        "schema_version": 1,
        "framework": "pytorch",
        "run_id": args.run_id,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "model": config,
    }
    (args.output / "pytorch_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"alignment_output={args.output}")
    print("framework=pytorch")
    print(f"device={device}")
    print("status=pass")


if __name__ == "__main__":
    main()
