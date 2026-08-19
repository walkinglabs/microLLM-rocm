import json
import math
import os
import subprocess
import unittest

import torch
import torch.nn.functional as F


def tensor(values, shape, requires_grad=False):
    return torch.tensor(values, dtype=torch.float32).reshape(shape).requires_grad_(requires_grad)


def rope(value, sequence_dim=1, position_offset=0, base=10000.0):
    width = value.shape[-1]
    positions = torch.arange(
        position_offset,
        position_offset + value.shape[sequence_dim],
        dtype=value.dtype,
        device=value.device,
    )
    frequencies = base ** (-torch.arange(0, width, 2, dtype=value.dtype) / width)
    angles = positions[:, None] * frequencies[None, :]
    view_shape = [1] * value.dim()
    view_shape[sequence_dim] = value.shape[sequence_dim]
    view_shape[-1] = width // 2
    cosine = torch.cos(angles).reshape(view_shape)
    sine = torch.sin(angles).reshape(view_shape)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def causal_softmax(scores):
    sequence = scores.shape[-1]
    future = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)


def record(mapping, name, value):
    detached = value.detach().to(torch.float32).cpu().clone()
    mapping[name] = (list(detached.shape), detached.reshape(-1))


def pytorch_references(actual):
    refs = {}
    record(refs, "fill", torch.full((2, 3), -1.25, dtype=torch.float32))
    left = tensor([1, -2, 3, 4, 0.5, -0.25], (2, 3))
    right = tensor([4, 5, -6, 2, 1.5, 0.25], (2, 3))
    record(refs, "add", left + right)
    record(refs, "multiply", left * right)
    record(refs, "scale", left * -0.25)

    matrix_left = tensor([1, 2, 3, 4, 5, 6], (2, 3))
    matrix_right = tensor([1, 2, 3, 4, 5, 6], (3, 2))
    record(refs, "matmul_2d", matrix_left @ matrix_right)
    record(refs, "matmul_readable", matrix_left @ matrix_right)
    record(
        refs,
        "matmul_3d",
        torch.matmul(
            tensor([1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1], (2, 2, 3)),
            tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], (2, 3, 2)),
        ),
    )

    embedding_weight = tensor([0, 1, 2, 3, 4, 5, 6, 7], (4, 2))
    indices = torch.tensor([2, 0, 2], dtype=torch.long)
    record(refs, "embedding", F.embedding(indices, embedding_weight))
    nonlinear = tensor([1000, 1000, 999, -2, 0, 2], (2, 3))
    record(refs, "softmax", torch.softmax(nonlinear, dim=-1))
    record(refs, "rms_norm", F.rms_norm(nonlinear, (3,), tensor([1, 0.5, 2], (3,)), 1.0e-5))
    record(refs, "silu", F.silu(left))
    record(refs, "swiglu", F.silu(left) * right)
    rope_input = tensor([1, 0, 0, 1, 1, 0, 0, 1], (1, 2, 1, 4))
    record(refs, "rope", rope(rope_input))
    logits = tensor([2, 1, 0, 100, -100, 0], (2, 3))
    targets = torch.tensor([0, -100], dtype=torch.long)
    record(refs, "cross_entropy", F.cross_entropy(logits, targets, ignore_index=-100))
    record(refs, "reduce_sum", torch.sum(left))
    record(refs, "broadcast_scalar", torch.tensor(2.5).expand(2, 3).clone())
    scores = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], (1, 3, 3))
    record(refs, "causal_softmax", causal_softmax(scores))
    record(refs, "repeat_interleave", torch.repeat_interleave(tensor([1, 2, 3, 4], (2, 2)), 2, 0))

    a = tensor([1, 2, 3, 4], (2, 2), True)
    b = tensor([5, 6, 7, 8], (2, 2), True)
    basic_output = a * b + a * 2.0
    basic_loss = basic_output.mean()
    basic_loss.backward()
    record(refs, "graph_basic_output", basic_output)
    record(refs, "graph_basic_loss", basic_loss)
    record(refs, "graph_basic_a_grad", a.grad)
    record(refs, "graph_basic_b_grad", b.grad)

    mat_left = tensor([1, 2, 3, 4, 5, 6], (2, 3), True)
    mat_right = tensor([1, 2, 3, 4, 5, 6], (3, 2), True)
    mat_seed = tensor([1, 2, 3, 4], (2, 2))
    ((mat_left @ mat_right) * mat_seed).sum().backward()
    record(refs, "graph_matmul_left_grad", mat_left.grad)
    record(refs, "graph_matmul_right_grad", mat_right.grad)

    embed_weight = tensor([0, 1, 2, 3, 4, 5, 6, 7], (4, 2), True)
    embed_seed = tensor([1, 2, 3, 4, 5, 6], (3, 2))
    (F.embedding(indices, embed_weight) * embed_seed).sum().backward()
    record(refs, "graph_embedding_weight_grad", embed_weight.grad)

    softmax_input = tensor([2, 1, 0, -1, 0, 1], (2, 3), True)
    softmax_seed = tensor([1, 2, 3, 4, 5, 6], (2, 3))
    (torch.softmax(softmax_input, -1) * softmax_seed).sum().backward()
    record(refs, "graph_softmax_input_grad", softmax_input.grad)

    norm_input = tensor([1, 2, 3, 4, 5, 6], (2, 3), True)
    norm_weight = tensor([1, 0.5, 2], (3,), True)
    norm_seed = tensor([1, -1, 2, -2, 3, -3], (2, 3))
    (F.rms_norm(norm_input, (3,), norm_weight, 1.0e-5) * norm_seed).sum().backward()
    record(refs, "graph_rms_input_grad", norm_input.grad)
    record(refs, "graph_rms_weight_grad", norm_weight.grad)

    silu_input = tensor([-2, -1, 0, 1, 2, 3], (2, 3), True)
    activation_seed = tensor([1, 2, 3, -1, -2, -3], (2, 3))
    (F.silu(silu_input) * activation_seed).sum().backward()
    record(refs, "graph_silu_input_grad", silu_input.grad)

    gate = tensor([-2, -1, 0, 1, 2, 3], (2, 3), True)
    up = tensor([1, 2, 3, 4, 5, 6], (2, 3), True)
    (F.silu(gate) * up * activation_seed).sum().backward()
    record(refs, "graph_swiglu_gate_grad", gate.grad)
    record(refs, "graph_swiglu_up_grad", up.grad)

    rope_value = tensor([1, 0, 0, 1, 1, 0, 0, 1], (1, 2, 1, 4), True)
    rope_seed = tensor([1, 2, 3, 4, -1, -2, -3, -4], (1, 2, 1, 4))
    (rope(rope_value) * rope_seed).sum().backward()
    record(refs, "graph_rope_input_grad", rope_value.grad)

    ce_logits = tensor([2, 1, 0, 100, -100, 0], (2, 3), True)
    (F.cross_entropy(ce_logits, targets, ignore_index=-100) * 0.75).backward()
    record(refs, "graph_cross_entropy_logits_grad", ce_logits.grad)

    causal_input = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], (1, 3, 3), True)
    causal_seed = tensor([1, 2, 3, -1, 0, 1, 2, -2, 0.5], (1, 3, 3))
    (causal_softmax(causal_input) * causal_seed).sum().backward()
    record(refs, "graph_causal_softmax_input_grad", causal_input.grad)

    repeat_input = tensor([1, 2, 3, 4], (2, 2), True)
    repeat_seed = tensor([1, 2, 3, 4, 5, 6, 7, 8], (4, 2))
    (torch.repeat_interleave(repeat_input, 2, 0) * repeat_seed).sum().backward()
    record(refs, "graph_repeat_input_grad", repeat_input.grad)

    view_input = tensor([0, 1, 2, 3, 4, 5], (2, 3), True)
    view_seed = tensor([1, 2, 3, 4, 5, 6], (3, 2))
    (view_input.transpose(0, 1).contiguous() * view_seed).sum().backward()
    record(refs, "graph_view_input_grad", view_input.grad)

    reshape_input = tensor([0, 1, 2, 3, 4, 5], (2, 3), True)
    reshape_seed = tensor([1, 2, 3, 4, 5, 6], (3, 2))
    (reshape_input.reshape(3, 2) * reshape_seed).sum().backward()
    record(refs, "graph_reshape_input_grad", reshape_input.grad)

    params = {}
    for case, (shape, values) in actual.items():
        if case.startswith("model_param:"):
            name = case.removeprefix("model_param:")
            parameter = values.reshape(shape).clone().requires_grad_(True)
            params[name] = parameter
            record(refs, case, parameter)

    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    model_targets = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    hidden = F.embedding(tokens, params["token_embedding.weight"])
    prefix = "blocks.0"
    normalized = F.rms_norm(hidden, (8,), params[f"{prefix}.attention_norm.weight"], 1.0e-5)
    flat = normalized.reshape(4, 8)
    query = (flat @ params[f"{prefix}.attention.q_proj.weight"]).reshape(1, 4, 2, 4).transpose(1, 2)
    key = (flat @ params[f"{prefix}.attention.k_proj.weight"]).reshape(1, 4, 1, 4).transpose(1, 2)
    value = (flat @ params[f"{prefix}.attention.v_proj.weight"]).reshape(1, 4, 1, 4).transpose(1, 2)
    query = rope(query, sequence_dim=2)
    key = rope(key, sequence_dim=2)
    key = torch.repeat_interleave(key, 2, dim=1)
    value = torch.repeat_interleave(value, 2, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(4.0)
    probabilities = causal_softmax(scores)
    context = torch.matmul(probabilities, value).transpose(1, 2).contiguous().reshape(4, 8)
    attention = (context @ params[f"{prefix}.attention.o_proj.weight"]).reshape(1, 4, 8)
    hidden = hidden + attention
    normalized = F.rms_norm(hidden, (8,), params[f"{prefix}.ffn_norm.weight"], 1.0e-5)
    flat = normalized.reshape(4, 8)
    gate = flat @ params[f"{prefix}.feed_forward.gate_proj.weight"]
    up = flat @ params[f"{prefix}.feed_forward.up_proj.weight"]
    feed_forward = (F.silu(gate) * up) @ params[f"{prefix}.feed_forward.down_proj.weight"]
    hidden = hidden + feed_forward.reshape(1, 4, 8)
    hidden = F.rms_norm(hidden, (8,), params["final_norm.weight"], 1.0e-5)
    model_logits = (hidden.reshape(4, 8) @ params["output_head.weight"]).reshape(1, 4, 8)
    model_loss = F.cross_entropy(model_logits.reshape(4, 8), model_targets.reshape(4))
    record(refs, "model_logits", model_logits)
    record(refs, "model_loss", model_loss)
    model_loss.backward()
    for name, parameter in params.items():
        record(refs, f"model_grad:{name}", parameter.grad)

    sgd_parameter = tensor([1.0, -2.0], (2,), True)
    sgd = torch.optim.SGD([sgd_parameter], lr=0.1, weight_decay=0.01)
    sgd_parameter.grad = tensor([0.5, -0.25], (2,))
    sgd.step()
    record(refs, "optimizer_sgd_parameter_step1", sgd_parameter)

    adam_parameter = tensor([1.0, -2.0], (2,), True)
    adam = torch.optim.AdamW(
        [adam_parameter], lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.1
    )
    adam_parameter.grad = tensor([0.5, -0.25], (2,))
    adam.step()
    record(refs, "optimizer_adamw_parameter_step1", adam_parameter)
    record(refs, "optimizer_adamw_first_moment_step1", adam.state[adam_parameter]["exp_avg"])
    record(refs, "optimizer_adamw_second_moment_step1", adam.state[adam_parameter]["exp_avg_sq"])
    adam_parameter.grad = tensor([-1.0, 2.0], (2,))
    adam.step()
    record(refs, "optimizer_adamw_parameter_step2", adam_parameter)
    record(refs, "optimizer_adamw_first_moment_step2", adam.state[adam_parameter]["exp_avg"])
    record(refs, "optimizer_adamw_second_moment_step2", adam.state[adam_parameter]["exp_avg_sq"])
    return refs


class OperatorParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        completed = subprocess.run(
            [os.environ["MICROLLM_OPERATOR_ORACLE"]],
            check=True,
            text=True,
            capture_output=True,
        )
        cls.actual = {}
        cls.rejections = {}
        cls.metadata = {}
        for line in completed.stdout.splitlines():
            item = json.loads(line)
            if "values" in item:
                cls.actual[item["name"]] = (
                    item["shape"],
                    torch.tensor(item["values"], dtype=torch.float32),
                )
            else:
                destination = cls.rejections if item["name"].startswith("invalid_") else cls.metadata
                destination[item["name"]] = item["bool"]
        cls.references = pytorch_references(cls.actual)

    def test_every_numeric_case_has_a_pytorch_reference_and_matching_shape(self):
        self.assertEqual(set(self.actual), set(self.references))
        for name, (expected_shape, _) in self.references.items():
            self.assertEqual(self.actual[name][0], expected_shape, name)

    def test_forward_and_backward_values_match_pytorch(self):
        looser = {
            "matmul_2d",
            "matmul_readable",
            "matmul_3d",
            "rms_norm",
            "graph_rms_input_grad",
            "graph_rms_weight_grad",
        }
        for name, (_, expected) in self.references.items():
            actual = self.actual[name][1]
            if name.startswith("model_grad:") or name in {"model_logits", "model_loss"}:
                tolerance = 2.0e-3
            else:
                tolerance = 3.0e-4 if name in looser else 3.0e-5
            torch.testing.assert_close(
                actual,
                expected.reshape(-1),
                atol=tolerance,
                rtol=tolerance,
                msg=lambda message, case=name: f"{case}: {message}",
            )

    def test_every_declared_invalid_shape_or_dtype_is_rejected(self):
        expected = {
            "invalid_add_shape",
            "invalid_multiply_shape",
            "invalid_scale_dtype",
            "invalid_matmul_inner",
            "invalid_embedding_weight",
            "invalid_softmax_dim",
            "invalid_rms_weight",
            "invalid_silu_dtype",
            "invalid_swiglu_shape",
            "invalid_rope_width",
            "invalid_cross_entropy_shape",
            "invalid_reduce_dtype",
            "invalid_broadcast_source",
            "invalid_causal_shape",
            "invalid_repeat_count",
            "invalid_embedding_backward_shape",
            "invalid_softmax_backward_shape",
            "invalid_rms_backward_shape",
            "invalid_silu_backward_shape",
            "invalid_swiglu_backward_shape",
            "invalid_rope_backward_width",
            "invalid_cross_entropy_backward_seed",
            "invalid_causal_backward_shape",
            "invalid_repeat_backward_shape",
        }
        self.assertEqual(set(self.rejections), expected)
        self.assertTrue(all(self.rejections.values()))

    def test_model_graph_snapshot_has_real_topology(self):
        self.assertEqual(self.metadata, {"model_graph_has_topology": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
