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
