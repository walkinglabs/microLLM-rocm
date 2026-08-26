import os
import unittest

import torch

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
                    left = (values.reshape(shape) * 0.03125).to(dtype)
                    right = (torch.flip(values, dims=(0,)).reshape(shape) * -0.015625).to(dtype)
                    actual_add = torch_ops.add(left, right)
                    actual_multiply = torch_ops.multiply(left, right)
                    if left.numel() != 0:
                        self.assertNotEqual(actual_add.data_ptr(), left.data_ptr())
                        self.assertNotEqual(actual_multiply.data_ptr(), right.data_ptr())
                    torch.testing.assert_close(actual_add, left + right, rtol=0, atol=0)
                    torch.testing.assert_close(actual_multiply, left * right, rtol=0, atol=0)

        with self.assertRaisesRegex(RuntimeError, "float32, float16, or bfloat16"):
            torch_ops.add(torch.ones(3, dtype=torch.int32),
                          torch.ones(3, dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, "dtypes must match"):
            torch_ops.multiply(torch.ones(3), torch.ones(3, dtype=torch.float16))
        with self.assertRaisesRegex(RuntimeError, "shapes must match"):
            torch_ops.add(torch.ones(3), torch.ones(4))
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch_ops.multiply(torch.ones(2, 3).transpose(0, 1), torch.ones(3, 2))

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

    def test_compile_fullgraph_uses_meta_contract(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")

        def function(left, right):
            return torch_ops.add(left, right) + torch_ops.multiply(left, right)

        compiled = torch.compile(function, backend="eager", fullgraph=True)
        devices = [torch.device("cpu")]
        if torch.version.hip and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            left = torch.arange(32, dtype=torch.float32, device=device)
            right = torch.full_like(left, 0.25)
            torch.testing.assert_close(compiled(left, right), function(left, right))

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "PyTorch ROCm unavailable")
    def test_rocm_add_uses_current_stream(self):
        device = torch.device("cuda")
        stream = torch.cuda.Stream()
        left = torch.arange(1024, dtype=torch.float32, device=device)
        right = torch.ones_like(left)
        with torch.cuda.stream(stream):
            actual = torch_ops.add(left, right)
        stream.synchronize()
        torch.testing.assert_close(actual, left + right)


if __name__ == "__main__":
    unittest.main(verbosity=2)
