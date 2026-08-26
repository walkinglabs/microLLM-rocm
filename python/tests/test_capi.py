import ctypes
import gc
import math
import unittest
import weakref

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

    def test_cpu_stream_routes_out_operator_and_event(self):
        stream = microllm.Stream("cpu")
        self.assertEqual(stream.device, (microllm.Device.CPU, 0))
        self.assertTrue(stream.non_blocking)
        self.assertTrue(stream.owning)
        self.assertEqual(stream.native_handle, 0)
        with self.assertRaises(ValueError):
            microllm.Stream.from_external(0)
        with self.assertRaises(microllm.MicroLLMError):
            microllm.Stream.from_external(1, "cpu")
        left = microllm.Tensor.from_f32([1, 2, 3, 4], (2, 2))
        right = microllm.Tensor.from_f32([5, 6, 7, 8], (2, 2))
        output = microllm.multiply(left, right)
        matmul_output = microllm.matmul(left, right)
        microllm.multiply_out(output, right, left, stream=stream)
        microllm.matmul_out(matmul_output, right, left, stream=stream)
        streamed_sum = microllm.add(left, right, stream=stream)
        streamed_product = microllm.matmul(left, right, stream=stream)
        streamed_softmax = microllm.softmax(left, stream=stream)
        marker = microllm.Event("cpu", enable_timing=False)
        marker.record(stream)
        marker.wait(stream)
        stream.synchronize()
        self.assertTrue(marker.ready())
        self.assertEqual(output.tolist(), [5, 12, 21, 32])
        self.assertEqual(matmul_output.tolist(), [23, 34, 31, 46])
        self.assertEqual(streamed_sum.tolist(), [6, 8, 10, 12])
        self.assertEqual(streamed_product.tolist(), [19, 22, 43, 50])
        self.assertAlmostEqual(sum(streamed_softmax.tolist()[:2]), 1.0, places=6)
        softmax_output = microllm.Tensor.from_f32([0, 0, 0, 0], (2, 2))
        microllm.softmax_out(softmax_output, left, stream=stream)
        softmax_values = softmax_output.tolist()
        self.assertAlmostEqual(sum(softmax_values[:2]), 1.0, places=6)
        self.assertAlmostEqual(sum(softmax_values[2:]), 1.0, places=6)
        norm_weight = microllm.Tensor.from_f32([1, 0.5], (2,))
        norm_output = microllm.Tensor.from_f32([0, 0, 0, 0], (2, 2))
        microllm.rms_norm_out(
            norm_output, left, norm_weight, stream=stream)
        norm_values = norm_output.tolist()
        for row in range(2):
            first, second = [1 + row * 2, 2 + row * 2]
            denominator = math.sqrt((first * first + second * second) / 2 + 1.0e-5)
            self.assertAlmostEqual(norm_values[row * 2], first / denominator, places=6)
            self.assertAlmostEqual(norm_values[row * 2 + 1],
                                   second * 0.5 / denominator, places=6)
        bf16_owner = (ctypes.c_uint16 * 4)(0, 0, 0, 0)
        bf16_output = microllm.Tensor.from_external(
            ctypes.addressof(bf16_owner), ctypes.sizeof(bf16_owner),
            (2, 2), (2, 1), dtype=microllm.DType.BFLOAT16,
            owner=bf16_owner)
        microllm.rms_norm_bf16_out(
            bf16_output, left, norm_weight, stream=stream)
        for actual, expected in zip(bf16_output.tolist(), norm_values):
            self.assertAlmostEqual(actual, expected, delta=0.01)
        swiglu_output = microllm.Tensor.from_f32([0, 0, 0, 0], (2, 2))
        microllm.swiglu_out(swiglu_output, left, right, stream=stream)
        for actual, gate, up in zip(
                swiglu_output.tolist(), left.tolist(), right.tolist()):
            self.assertAlmostEqual(actual,
                                   gate / (1 + math.exp(-gate)) * up,
                                   places=5)
        query = microllm.Tensor.from_f32([0] * 8, (1, 2, 2, 2))
        key = microllm.Tensor.from_f32([0] * 4, (1, 1, 2, 2))
        value = microllm.Tensor.from_f32([1, 2, 3, 4], (1, 1, 2, 2))
        attention_output = microllm.Tensor.from_f32([0] * 8, (1, 2, 2, 2))
        scaled_query = microllm.Tensor.from_f32([0] * 8, (1, 2, 2, 2))
        expanded_kv = microllm.Tensor.from_f32([0] * 8, (1, 2, 2, 2))
        probabilities = microllm.Tensor.from_f32([0] * 8, (1, 2, 2, 2))
        microllm.causal_gqa_attention_out(
            attention_output, scaled_query, expanded_kv, probabilities,
            query, key, value, repeats=2, scale=0.5, stream=stream)
        self.assertEqual(attention_output.tolist(),
                         [1, 2, 2, 3, 1, 2, 2, 3])
        with self.assertRaises(microllm.MicroLLMError):
            microllm.causal_gqa_attention_out(
                query, scaled_query, expanded_kv, probabilities,
                query, key, value, repeats=2, scale=0.5, stream=stream)
        embedding_weight = microllm.Tensor.from_f32(
            [0, 1, 2, 3, 4, 5], (3, 2))
        embedding_indices = microllm.Tensor.from_i32([2, 0, 1], (3,))
        embedding_output = microllm.Tensor.from_f32([0] * 6, (3, 2))
        microllm.embedding_out(
            embedding_output, embedding_weight, embedding_indices,
            stream=stream)
        self.assertEqual(embedding_output.tolist(), [4, 5, 0, 1, 2, 3])
        rope_input = microllm.Tensor.from_f32(
            [1, 0, 0, 1, 1, 0, 0, 1], (1, 2, 1, 4))
        rope_output = microllm.Tensor.from_f32([0] * 8, (1, 2, 1, 4))
        microllm.rope_out(rope_output, rope_input, stream=stream)
        self.assertEqual(rope_output.tolist()[:4], [1, 0, 0, 1])
        logits = microllm.Tensor.from_f32([2, 1, 0, 0, 1, 2], (2, 3))
        targets = microllm.Tensor.from_i32([0, 2], (2,))
        loss_output = microllm.Tensor.from_f32([0], ())
        loss_workspace = microllm.Tensor.from_f32([0] * 4, (2, 2))
        microllm.cross_entropy_out(
            loss_output, loss_workspace, logits, targets, stream=stream)
        expected_loss = math.log(math.exp(2) + math.exp(1) + 1) - 2
        self.assertAlmostEqual(loss_output.tolist()[0], expected_loss, places=6)

    def test_external_tensor_is_zero_copy_non_owning_and_strict(self):
        left_owner = (ctypes.c_float * 4)(1, 2, 3, 4)
        right_owner = (ctypes.c_float * 4)(5, 6, 7, 8)
        output_owner = (ctypes.c_float * 4)(0, 0, 0, 0)
        left_ref = weakref.ref(left_owner)
        left = microllm.Tensor.from_external(
            ctypes.addressof(left_owner), ctypes.sizeof(left_owner),
            (2, 2), (2, 1), owner=left_owner)
        right = microllm.Tensor.from_external(
            ctypes.addressof(right_owner), ctypes.sizeof(right_owner),
            (2, 2), (2, 1), owner=right_owner)
        output = microllm.Tensor.from_external(
            ctypes.addressof(output_owner), ctypes.sizeof(output_owner),
            (2, 2), (2, 1), owner=output_owner)
        self.assertFalse(left.owning)
        self.assertEqual(left.data_ptr, ctypes.addressof(left_owner))
        self.assertEqual(left.storage_bytes, ctypes.sizeof(left_owner))
        stream = microllm.Stream("cpu")
        microllm.add_out(output, left, right, stream=stream)
        self.assertEqual(list(output_owner), [6, 8, 10, 12])
        left_owner[0] = 10
        microllm.add_out(output, left, right, stream=stream)
        self.assertEqual(list(output_owner), [15, 8, 10, 12])
        with self.assertRaises(microllm.MicroLLMError):
            microllm.Tensor.from_external(
                ctypes.addressof(left_owner), 4, (2, 2), (2, 1),
                owner=left_owner)
        noncontiguous = microllm.Tensor.from_external(
            ctypes.addressof(left_owner), ctypes.sizeof(left_owner),
            (2, 2), (1, 2), owner=left_owner)
        with self.assertRaises(microllm.MicroLLMError):
            microllm.add_out(output, noncontiguous, right, stream=stream)
        noncontiguous.close()
        with self.assertRaises(ValueError):
            microllm.Tensor.from_external(
                ctypes.addressof(left_owner), ctypes.sizeof(left_owner),
                (2, 2), (2,), owner=left_owner)
        with self.assertRaises(microllm.MicroLLMError):
            microllm.Tensor.from_external(
                ctypes.addressof(left_owner), ctypes.sizeof(left_owner),
                (2, 2), (2, -1), owner=left_owner)
        int_owner = (ctypes.c_int32 * 2)(7, 11)
        int_tensor = microllm.Tensor.from_external(
            ctypes.addressof(int_owner), ctypes.sizeof(int_owner),
            (2,), (1,), dtype=microllm.DType.INT32, owner=int_owner)
        self.assertEqual(int_tensor.tolist(), [7, 11])
        with self.assertRaises(microllm.MicroLLMError):
            microllm.add_out(int_tensor, int_tensor, int_tensor, stream=stream)
        low_cases = (
            (microllm.DType.FLOAT16, (0x3C00, 0x4000),
             (0x4200, 0x4400)),
            (microllm.DType.BFLOAT16, (0x3F80, 0x4000),
             (0x4040, 0x4080)),
        )
        for dtype, left_bits, right_bits in low_cases:
            low_left_owner = (ctypes.c_uint16 * 2)(*left_bits)
            low_right_owner = (ctypes.c_uint16 * 2)(*right_bits)
            low_output_owner = (ctypes.c_uint16 * 2)(0, 0)
            low_left = microllm.Tensor.from_external(
                ctypes.addressof(low_left_owner), ctypes.sizeof(low_left_owner),
                (2,), (1,), dtype=dtype, owner=low_left_owner)
            low_right = microllm.Tensor.from_external(
                ctypes.addressof(low_right_owner), ctypes.sizeof(low_right_owner),
                (2,), (1,), dtype=dtype, owner=low_right_owner)
            low_output = microllm.Tensor.from_external(
                ctypes.addressof(low_output_owner), ctypes.sizeof(low_output_owner),
                (2,), (1,), dtype=dtype, owner=low_output_owner)
            microllm.multiply_out(
                low_output, low_left, low_right, stream=stream)
            self.assertEqual(low_output.dtype, dtype)
            self.assertEqual(low_output.tolist(), [3.0, 8.0])
        del left_owner
        gc.collect()
        self.assertIsNotNone(left_ref())
        left.close()
        gc.collect()
        self.assertIsNone(left_ref())

    def test_optional_hip_roundtrip(self):
        if microllm.hip_device_count() == 0:
            self.skipTest("no visible HIP device")
        tensor = microllm.Tensor.from_f32([1, 2, 3, 4], (2, 2)).to("hip:0")
        self.assertEqual(tensor.device, (microllm.Device.HIP, 0))
        start = microllm.Event("hip:0")
        finish = microllm.Event("hip:0")
        stream = microllm.Stream("hip:0")
        external = microllm.Stream.from_external(
            stream.native_handle, device="hip:0")
        self.assertTrue(stream.owning)
        self.assertFalse(external.owning)
        self.assertEqual(external.native_handle, stream.native_handle)
        tensor_alias = microllm.Tensor.from_external(
            tensor.data_ptr, tensor.storage_bytes, tensor.shape, (2, 1),
            device="hip:0", owner=tensor)
        output_owner = tensor + tensor
        output_alias = microllm.Tensor.from_external(
            output_owner.data_ptr, output_owner.storage_bytes,
            output_owner.shape, (2, 1), device="hip:0", owner=output_owner)
        with self.assertRaises(microllm.MicroLLMError):
            microllm.add(tensor, tensor, stream=microllm.Stream("cpu"))
        start.record(external)
        microllm.add_out(output_alias, tensor_alias, tensor_alias,
                         stream=external)
        finish.record(external)
        finish.synchronize()
        external.close()
        stream.synchronize()
        self.assertTrue(finish.ready())
        self.assertGreaterEqual(finish.elapsed_ms_since(start), 0.0)
        self.assertEqual(output_owner.tolist(), [2, 4, 6, 8])
        tensor_alias.close()
        output_alias.close()
        self.assertEqual(tensor.tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
