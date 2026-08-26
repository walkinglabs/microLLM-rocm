# PyTorch Tensor不搬家，microLLM借指针计算

日期：2026-08-26
状态：FP32 contiguous零复制add已验收

## 描述符必须说清什么

裸指针不知道能访问多少字节，也不知道shape、stride、dtype和设备。新接口要求调用方一次提供：

```text
data_ptr + storage_bytes + shape + element_strides + dtype + device
```

C++用`Storage::from_external`和`Tensor::from_storage`复用已有布局检查。`ml_tensor_destroy`只销毁
wrapper。Python `Tensor.from_external(..., owner=torch_tensor)`额外持有owner，避免外部Tensor先被
垃圾回收。

## 第一条严格算子路径

`ml_add_out_on_stream`/Python `add_out`直接调用low-level TensorView算子，要求：

- FP32；
- contiguous element-stride；
- 三个shape和device完全相同；
- caller提供output和Stream；
- 无输出分配，无fallback copy。

非连续view不是“不能描述”，而是当前add Kernel不支持。测试要求它在launch前失败，防止Agent为了
让用例通过而插入`contiguous()`。

## PyTorch ROCm三进程证据

每轮包装三张4096×1024 Tensor，指针逐个相同且ownership为false。microLLM先写3，再读取Torch
把left改成10后的值并写12；两个完整Torch输出Max均为0。三轮暴露144MiB，wrapper payload copy为
0字节，共384次add提交。

owner weakref在wrapper存在时存活，close后释放；output wrapper销毁后Torch继续在原指针写/读42。
短storage和non-contiguous路径均3/3失败。

![Zero-copy Tensor](../../benchmarks/results/2026-08-26-pytorch-zero-copy-tensor/zero-copy-tensor.svg)

## 不能写成什么

这不是“PyTorch集成已完成”。当前只覆盖FP32 contiguous add，且rocprof混合进程仍被LLVM冲突
阻塞。下一步要逐dtype/算子扩大caller-owned输出，并将同一描述符接入可选PyTorch Custom Op；每个
扩展仍需指针、生命周期、完整输出和无隐式copy证据。
