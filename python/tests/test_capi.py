import math
import unittest

import microllm


class TensorTest(unittest.TestCase):
    def test_shape_dtype_add_and_matmul(self):
        left = microllm.Tensor.from_f32([1, 2, 3, 4], (2, 2))
        right = microllm.Tensor.from_f32([5, 6, 7, 8], (2, 2))
        self.assertEqual(left.shape, (2, 2))
        self.assertEqual(left.dtype, microllm.DType.FLOAT32)
        self.assertEqual((left + right).tolist(), [6, 8, 10, 12])
        self.assertEqual((left @ right).tolist(), [19, 22, 43, 50])

    def test_softmax_and_int32_roundtrip(self):
        probabilities = microllm.softmax(
            microllm.Tensor.from_f32([1000, 1000, 999, 1, 2, 3], (2, 3))
        ).tolist()
        self.assertAlmostEqual(sum(probabilities[:3]), 1.0, places=6)
        self.assertAlmostEqual(sum(probabilities[3:]), 1.0, places=6)
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        tokens = microllm.Tensor.from_i32([1, 2, 3], (3,))
        self.assertEqual(tokens.tolist(), [1, 2, 3])

    def test_python_and_engine_errors_are_visible(self):
        with self.assertRaises(ValueError):
            microllm.Tensor.from_f32([1, 2], (3,))
        left = microllm.Tensor.from_f32([1, 2], (2,))
        right = microllm.Tensor.from_f32([1, 2, 3], (3,))
        with self.assertRaises(microllm.MicroLLMError):
            _ = left + right

    def test_event_default_stream_lifecycle(self):
        event = microllm.Event("cpu", enable_timing=False)
        self.assertEqual(event.device, (microllm.Device.CPU, 0))
        self.assertFalse(event.timing_enabled)
        self.assertFalse(event.ready())
        event.record_default_stream()
        self.assertTrue(event.ready())
        event.synchronize()
        with self.assertRaises(microllm.MicroLLMError):
            event.elapsed_ms_since(event)

    def test_optional_hip_roundtrip(self):
        if microllm.hip_device_count() == 0:
            self.skipTest("no visible HIP device")
        tensor = microllm.Tensor.from_f32([1, 2, 3, 4], (2, 2)).to("hip:0")
        self.assertEqual(tensor.device, (microllm.Device.HIP, 0))
        start = microllm.Event("hip:0")
        finish = microllm.Event("hip:0")
        start.record_default_stream()
        result = tensor + tensor
        finish.record_default_stream()
        finish.synchronize()
        self.assertTrue(finish.ready())
        self.assertGreaterEqual(finish.elapsed_ms_since(start), 0.0)
        self.assertEqual(result.tolist(), [2, 4, 6, 8])


if __name__ == "__main__":
    unittest.main()
