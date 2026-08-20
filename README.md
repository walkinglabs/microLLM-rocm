# microLLM-rocm 初学者课程

这是 `microLLM-rocm` 的**纯课程分支**。这里解释框架为什么这样设计，安排
N0–N10 Notebook 和 PA0–PA2 作业，但不复制框架源码、测试、构建产物或实测结果。

```text
main                       真实 C++/HIP 引擎、测试、Benchmark、开发记录
tutorial/beginner-course   课程正文、作业、学习用小数据、课程结构校验
```

框架的唯一真实来源是
[`main`](https://github.com/walkinglabs/microLLM-rocm/tree/main)。课程中出现的命令
都在 `main` 工作树运行；这样课程不会悄悄使用一份过时的引擎副本。

## 从哪里开始

第一次接触框架内部结构时，按下面顺序阅读：

1. [把一串数字变成会学习的小模型](docs/DESIGN_FOR_BEGINNERS.zh-CN.md)
2. [课程地图](notebooks/README.md)
3. [N0：从数组到 Storage 与 Tensor](notebooks/N0_storage_tensor.md)
4. [任务契约工作单](docs/TASK_CONTRACT.md)
5. [算子 shape、精度和失败契约](docs/OPERATOR_CONTRACTS.zh-CN.md)

## 准备课程和引擎

推荐使用两个相邻目录。第一个目录固定在 `main`，第二个目录只阅读课程：

```bash
git clone --branch main \
  https://github.com/walkinglabs/microLLM-rocm.git microLLM-rocm-engine
git clone --branch tutorial/beginner-course \
  https://github.com/walkinglabs/microLLM-rocm.git microLLM-rocm-course

cd microLLM-rocm-course
export MICROLLM_ENGINE_DIR="$(cd ../microLLM-rocm-engine && pwd)"
```

检查变量指向真正的框架分支：

```bash
test "$(git -C "$MICROLLM_ENGINE_DIR" branch --show-current)" = main
```

然后按照 `main` 的
[构建说明](https://github.com/walkinglabs/microLLM-rocm/blob/main/docs/dev/build.md)
完成 CPU、HIP 或 RCCL 构建。课程只引用这些构建，不维护另一套 CMake 工程。

## 学习主线

```text
数组和指针
  ↓
Storage：谁保管内存
  ↓
Tensor：怎样解释 shape、stride、dtype 和 device
  ↓
CPU reference：最容易检查的正确答案
  ↓
HIP Kernel：让 AMD GPU 计算同一道题
  ↓
Autograd：记录来路并计算梯度
  ↓
Transformer：把算子连接成语言模型
  ↓
训练、Checkpoint 和 KV Cache 生成
  ↓
性能测量和多卡训练
  ↓
真实 Hugging Face 权重与低精度
  ↓
RCCL 多卡训练和证据图集
```

课程不是只读文章。每章都要求：先预测、运行旧办法、复现一个失败、写任务
契约、运行 `main` 的测试、审查改动，并保留一个尚未解决的边界。

## 分支中允许出现什么

| 路径 | 内容 |
|---|---|
| `notebooks/` | N0–N10 连续课程 |
| `pa/` | PA0–PA2 作业说明和极小独立练习 |
| `docs/` | 初学者设计、算子契约、任务工作单 |
| `data/` | 可以直接审阅的小型教学数据和登记说明 |
| `course_tools/` | 只检查课程目录和 Markdown 链接的工具 |

下面这些属于 `main`，不得再次放进课程分支：

```text
src/ include/ tests/ apps/ examples/ bindings/ benchmarks/ CMakeLists.txt
```

## 课程校验

校验器不会编译框架。它只证明课程文件齐全、内部 Markdown 链接有效，并且
课程分支没有重新引入引擎目录：

```bash
python3 course_tools/validate_course.py
```

框架的数值、梯度、单卡和多卡结果必须在 `main` 重新运行。课程不能把文字中的
示例输出当作当前硬件证据。

## 作业

- [PA0：手算一次参数更新](pa/PA0/README.md)
- [PA1：提交一个可复现的性能失败](pa/PA1/README.md)
- [PA2：从稳定失败提出下一版系统](pa/PA2/README.md)

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。
