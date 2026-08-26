import os
import unittest

import torch
import torch.nn.functional as F

from microllm import torch_ops


class TorchOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_ops.load_library(os.environ["MICROLLM_TORCH_OP_LIBRARY"])

    def test_cpu_add_and_multiply(self):
        left = torch.tensor([1.0, 2.0, 3.0])
        right = torch.tensor([4.0, 5.0, 6.0])
        torch.testing.assert_close(torch_ops.add(left, right), left + right)
        torch.testing.assert_close(torch_ops.multiply(left, right), left * right)

    def test_dtype_shape_and_error_matrix(self):
        devices = [torch.device("cpu")]
        if torch.version.hip and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for shape in ((0,), (1,), (3, 5), (2, 3, 4), (65537,)):
                    values = torch.arange(
                        max(1, torch.tensor(shape).prod().item()),
                        dtype=torch.float32, device=device)[:torch.tensor(shape).prod().item()]
                    left = ((values % 251).reshape(shape) * 0.03125 - 2).to(dtype)
                    right = ((torch.flip(values, dims=(0,)) % 127).reshape(shape) *
                             -0.015625 + 1).to(dtype)
                    actual_add = torch_ops.add(left, right)
                    actual_multiply = torch_ops.multiply(left, right)
                    actual_softmax = torch_ops.softmax(left)
                    caller_softmax = torch.empty_like(left)
                    returned_softmax = torch_ops.softmax_out(left, caller_softmax)
                    actual_swiglu = torch_ops.swiglu(left, right)
                    if left.numel() != 0:
                        self.assertNotEqual(actual_add.data_ptr(), left.data_ptr())
                        self.assertNotEqual(actual_multiply.data_ptr(), right.data_ptr())
                        self.assertNotEqual(actual_softmax.data_ptr(), left.data_ptr())
                        self.assertNotEqual(actual_swiglu.data_ptr(), left.data_ptr())
                    torch.testing.assert_close(actual_add, left + right, rtol=0, atol=0)
                    torch.testing.assert_close(actual_multiply, left * right, rtol=0, atol=0)
                    softmax_tolerance = (2.0e-6 if dtype == torch.float32 else
                                         5.0e-4 if dtype == torch.float16 else 4.0e-3)
                    torch.testing.assert_close(
                        actual_softmax, torch.softmax(left, dim=-1),
                        rtol=0, atol=softmax_tolerance)
                    self.assertEqual(returned_softmax.data_ptr(), caller_softmax.data_ptr())
                    torch.testing.assert_close(
                        caller_softmax, torch.softmax(left, dim=-1),
                        rtol=0, atol=softmax_tolerance)
                    tolerance = (1.0e-6 if dtype == torch.float32 else
                                 4.0e-3 if dtype == torch.float16 else 6.25e-2)
                    torch.testing.assert_close(
                        actual_swiglu, F.silu(left) * right,
                        rtol=0, atol=tolerance)

        with self.assertRaisesRegex(RuntimeError, "float32, float16, or bfloat16"):
            torch_ops.add(torch.ones(3, dtype=torch.int32),
                          torch.ones(3, dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, "dtypes must match"):
            torch_ops.multiply(torch.ones(3), torch.ones(3, dtype=torch.float16))
        with self.assertRaisesRegex(RuntimeError, "shapes must match"):
            torch_ops.add(torch.ones(3), torch.ones(4))
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch_ops.multiply(torch.ones(2, 3).transpose(0, 1), torch.ones(3, 2))
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch_ops.softmax(torch.ones(2, 3).transpose(0, 1))
        with self.assertRaisesRegex(RuntimeError, "at least one dimension"):
            torch_ops.softmax(torch.tensor(1.0))
        with self.assertRaisesRegex(RuntimeError, "dtypes must match"):
            torch_ops.softmax_out(
                torch.ones(2, 3), torch.empty(2, 3, dtype=torch.float16))
        with self.assertRaisesRegex(RuntimeError, "shapes must match"):
            torch_ops.softmax_out(torch.ones(2, 3), torch.empty(3, 2))
        alias = torch.ones(2, 3)
        with self.assertRaisesRegex(RuntimeError, "must not alias"):
            torch_ops.softmax_out(alias, alias)
        with self.assertRaisesRegex(RuntimeError, "output must be contiguous"):
            torch_ops.softmax_out(torch.ones(2, 3), torch.empty(3, 2).transpose(0, 1))
        with self.assertRaisesRegex(RuntimeError, "shapes must match"):
            torch_ops.swiglu(torch.ones(3), torch.ones(4))

    def test_autograd_branch_matches_pytorch(self):
        devices = [torch.device("cpu")]
        if torch.version.hip and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                left = torch.linspace(-1, 1, 32, device=device, dtype=dtype).requires_grad_()
                right = torch.linspace(2, -2, 32, device=device, dtype=dtype).requires_grad_()
                loss = (torch_ops.add(left, right) +
                        torch_ops.multiply(left, right)).sum()
                loss.backward()
                torch.testing.assert_close(left.grad, 1 + right.detach(), rtol=0, atol=0)
                torch.testing.assert_close(right.grad, 1 + left.detach(), rtol=0, atol=0)

                native_left = left.detach().clone().requires_grad_()
                native_right = right.detach().clone().requires_grad_()
                native_loss = (F.silu(native_left) * native_right).sum()
                native_loss.backward()
                custom_left = left.detach().clone().requires_grad_()
                custom_right = right.detach().clone().requires_grad_()
                torch_ops.swiglu(custom_left, custom_right).sum().backward()
                tolerance = (2.0e-6 if dtype == torch.float32 else
                             4.0e-3 if dtype == torch.float16 else 6.25e-2)
                torch.testing.assert_close(
                    custom_left.grad, native_left.grad, rtol=0, atol=tolerance)
                torch.testing.assert_close(
                    custom_right.grad, native_right.grad, rtol=0, atol=tolerance)

                native_softmax_input = torch.linspace(
                    -2, 2, 32, device=device, dtype=dtype).reshape(2, 16).requires_grad_()
                custom_softmax_input = native_softmax_input.detach().clone().requires_grad_()
                seed = torch.linspace(
                    -1, 1, 32, device=device, dtype=dtype).reshape(2, 16)
                torch.softmax(native_softmax_input, dim=-1).backward(seed)
                torch_ops.softmax(custom_softmax_input).backward(seed)
                softmax_tolerance = (3.0e-6 if dtype == torch.float32 else
                                     1.0e-3 if dtype == torch.float16 else 8.0e-3)
                torch.testing.assert_close(
                    custom_softmax_input.grad, native_softmax_input.grad,
                    rtol=0, atol=softmax_tolerance)
                with self.assertRaisesRegex(RuntimeError, "inference-only"):
                    torch_ops.softmax_out(
                        custom_softmax_input, torch.empty_like(custom_softmax_input))

    def test_compile_fullgraph_uses_meta_contract(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")

        def function(left, right):
            return (torch_ops.add(left, right) +
                    torch_ops.multiply(left, right) +
                    torch_ops.swiglu(left, right) +
                    torch_ops.softmax(left))

        compiled = torch.compile(function, backend="eager", fullgraph=True)
        devices = [torch.device("cpu")]
        if torch.version.hip and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            left = torch.arange(32, dtype=torch.float32, device=device)
            right = torch.full_like(left, 0.25)
            torch.testing.assert_close(compiled(left, right), function(left, right))

    def test_swiglu_scalar_seed_matches_expanded_gradient(self):
        devices = [torch.device("cpu")]
        if torch.version.hip and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            gate = torch.linspace(-2, 2, 1027, device=device)
            up = torch.linspace(1, -1, 1027, device=device)
            seed = torch.tensor(0.5, device=device)
            actual = torch.ops.microllm.swiglu_backward_scalar_seed(
                gate, up, seed)
            expected_gate = gate.detach().clone().requires_grad_()
            expected_up = up.detach().clone().requires_grad_()
            (F.silu(expected_gate) * expected_up).backward(
                torch.full_like(gate, 0.5))
            torch.testing.assert_close(
                actual[0], expected_gate.grad, rtol=0, atol=3.0e-6)
            torch.testing.assert_close(
                actual[1], expected_up.grad, rtol=0, atol=3.0e-6)

            routed_gate = gate.detach().clone().requires_grad_()
            routed_up = up.detach().clone().requires_grad_()
            seen_layout = []
            output = torch_ops.swiglu(routed_gate, routed_up)
            output.register_hook(
                lambda gradient: seen_layout.append(
                    (gradient.stride(), gradient.untyped_storage().nbytes())))
            output.sum().backward()
            self.assertEqual(seen_layout, [((0,), 4)])
            torch.testing.assert_close(
                routed_gate.grad, expected_gate.grad * 2, rtol=0, atol=3.0e-6)
            torch.testing.assert_close(
                routed_up.grad, expected_up.grad * 2, rtol=0, atol=3.0e-6)

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "PyTorch ROCm unavailable")
    def test_rocm_add_uses_current_stream(self):
        device = torch.device("cuda")
        stream = torch.cuda.Stream()
        left = torch.arange(1024, dtype=torch.float32, device=device)
        right = torch.ones_like(left)
        with torch.cuda.stream(stream):
            actual = torch_ops.add(left, right)
            actual_softmax = torch_ops.softmax(left.reshape(8, 128))
            caller_softmax = torch.empty(8, 128, device=device)
            returned_softmax = torch_ops.softmax_out(
                left.reshape(8, 128), caller_softmax)
        stream.synchronize()
        torch.testing.assert_close(actual, left + right)
        torch.testing.assert_close(
            actual_softmax, torch.softmax(left.reshape(8, 128), dim=-1),
            rtol=0, atol=2.0e-6)
        self.assertEqual(returned_softmax.data_ptr(), caller_softmax.data_ptr())
        torch.testing.assert_close(
            caller_softmax, torch.softmax(left.reshape(8, 128), dim=-1),
            rtol=0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
