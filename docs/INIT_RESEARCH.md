# microLLM-rocm 初始化调研报告

> 历史说明：本文记录最初的 N0–N9 脚手架和实现前假设。当前架构与
> N0–N8 路线以 `docs/ARCHITECTURE.md` 和 `docs/development/ROADMAP.md` 为准。

> 生成时间: 2025-08
> 环境: Linux x86_64, 4× AMD MI300X (gfx942), ROCm 7.13, HIP 7.13, CMake 3.31, C++20

---

## 1. 参考项目调研

### 1.1 llama.cpp

**定位**: 高性能 C++ LLM 推理引擎, 支持 CPU / CUDA / Metal / Vulkan / ROCm

**目录结构 (简化)**:
```
llama.cpp/
├── src/                    # 核心实现
│   ├── llama.cpp/h         # 模型加载/推理主逻辑
│   ├── ggml.c/h            # 张量库 (底层)
│   ├── ggml-alloc.c/h      # 内存分配器
│   ├── ggml-backend.c/h    # 后端抽象 (CPU/CUDA/Metal)
│   ├── ggml-cuda/          # CUDA 算子
│   ├── ggml-hip/           # HIP 算子
│   └── llama-vocab.cpp     # 分词器
├── common/                 # 命令行工具共用代码
├── examples/               # 示例 (main, server, benchmark...)
├── ggml/src/               # ggml 库核心
└── tests/
```

**核心抽象**:
- `ggml_tensor`: 最核心结构体 — shape, stride, data, 类型, 后端
- `ggml_context`: 计算图上下文, 管理张量生命周期
- `ggml_gallocr`: 图级内存分配器 (先计算大小, 一次分配)
- `ggml_backend`: 后端接口 (CPU, CUDA, Metal 各自实现)
- GGUF 格式: 模型序列化, 支持量化 (Q4_0, Q5_K, Q8_0...)

**关键技术决策**:
- 纯 C 实现, 极简, 无 STL 依赖 → 移植性强
- 计算图构建 + 一次分配策略 → 内存效率高
- 量化优先设计 → 4-bit 模型可在 CPU 上运行
- 后端抽象层使同一代码跑在不同硬件

**可借鉴之处**:
- ✅ 后端抽象: 我们的 `Device` 概念类似, 但 C++ RAII 更安全
- ✅ 内存分配策略: 先规划再分配 (N0 阶段实现)
- ✅ 计算图思路: 为 N5 autograd 做铺垫
- ⚠️ llama.cpp 不做训练, 没有反向传播

---

### 1.2 ggml (张量库)

**定位**: llama.cpp 底层的张量计算库, 提供 ops + 内存管理

**核心设计**:
- `ggml_tensor` 结构体: `{type, backend, buffer, ne[], nb[], op, src[], grad, ...}`
- 所有算子用函数指针风格的 ops 表: `ggml_op_name[]`, `ggml_compute_forward[]`
- 用 `ggml_new_tensor()` 创建, `ggml_dup_tensor()` 克隆形状
- 自定义分配器: `ggml_allocr` 支持从预分配 buffer 中切分

**可借鉴之处**:
- ✅ ops 表设计思路 → 我们的算子注册表
- ✅ 分配器与张量分离 → Storage 独立管理
- ⚠️ 纯 C 风格, 不适合教学 (缺乏类型安全)

---

### 1.3 nanoGPT (Karpathy)

**定位**: 最简 GPT 训练实现, Python/PyTorch, 教学向

**目录结构**:
```
nanoGPT/
├── model.py       # GPT 模型定义 (~200 行)
├── train.py       # 训练循环
├── config/
│   ├── train_gpt2.py
│   └── train_shakespeare.py
├── data/          # 数据集下载脚本
└── scripts/
```

**核心设计**:
- 单文件 GPT: `GPTConfig` + `CausalSelfAttention` + `MLP` + `Block` + `GPT`
- 配置驱动: 不同规模的模型只需改 config dict
- 简洁的训练循环: loss → backward → step → log

**可借鉴之处**:
- ✅ 模型配置类设计 → 我们的 `ModelConfig`
- ✅ 教学节奏: 先 Shakespeare, 再 GPT-2 → 我们也是先小后大
- ✅ 模块拆分粒度: Attention → Block → Model → 适合逐层教学
- ⚠️ Python 魔法太多, 不适合 C++ 照搬

---

### 1.4 tinygrad

**定位**: 极简深度学习框架, 编译器思路, Python

**核心架构**:
```
tinygrad/
├── tinygrad/
│   ├── tensor.py       # Lazy Tensor (核心)
│   ├── ops/            # 算子定义
│   ├── runtime/        # 执行后端 (LLVM, CUDA, Metal, HIP...)
│   ├── engine/         # LazyExecutor, 调度/优化
│   └── nn/             # 高层模块
├── test/
└── docs/
```

**核心技术决策**:
- **Lazy Evaluation**: Tensor 不立即计算, 构建计算图后统一执行
- **ShapeTracker**: 用表达式树表示多维索引 (取代 stride)
- **Buffer**: 底层存储抽象, 类似我们的 Storage
- **Kernel**: 统一 kernel 抽象, 用 C-style 代码生成 → 运行时编译

**可借鉴之处**:
- ✅ Lazy execution 模型 → 我们的 Autograd 用类似思路
- ✅ Buffer/Storage 分离 → 与我们的设计一致
- ✅ 后端抽象: 一个算子多种实现 → CPU vs HIP 双路径
- ⚠️ 编译器太重, 不适合本课程
- ⚠️ Python 代码生成对 C++ 无直接参考

---

### 1.5 llm.c (Karpathy)

**定位**: 单文件 C 语言 GPT-2 训练, 零依赖

**目录结构**:
```
llm.c/
├── train_gpt2.c      # 单文件! ~800 行
├── dev/              # 实验性代码
├── doc/
└── README.md
```

**核心设计**:
- 所有内容在一个 .c 文件: 参数加载 → 前向传播 → 损失 → 反向传播 → 优化器更新
- 手写 GEMM (矩阵乘法) + 自定义反向传播
- 直接操作 float* 指针, 无张量抽象
- 使用 `mmap` 加载 GPT-2 权重

**关键技术决策**:
- 极致极简: 证明 LLM 训练不需要大框架
- 手动内存管理: `malloc` + `free`
- 串行训练 (单 GPU/单线程)
- 直接用 OpenMP 并行化矩阵运算

**可借鉴之处**:
- ✅ 证明了「从零实现训练」的可行性
- ✅ 前向+反向在同一文件的教学价值 → 我们按模块拆分, 但保持同样简洁
- ✅ GPT-2 的权重布局可作为参考
- ⚠️ 无张量抽象 → 难以扩展
- ⚠️ 无自动求导 → 硬编码反向传播

---

### 1.6 其他相关项目

#### candle (HuggingFace, Rust)
- **特点**: Rust + CUDA, 极简推理引擎
- **借鉴**: 模块化设计, 配置驱动的模型定义

#### MLC-LLM (Apache TVM)
- **特点**: 编译器驱动, 支持 ROCm
- **借鉴**: 量化 (GPTQ, AWQ) 的工程实现

#### PyTorch 源码 (ATen)
- **特点**: C++17, `c10::TensorImpl` 是 Storage+Tensor 的参考实现
- **借鉴**: `DispatchKey` 机制 (CPU/CUDA 分发), `AutogradMeta`

---

## 2. 技术栈选择及理由

### 2.1 构建系统: CMake ≥ 3.25

| 选项 | 优点 | 缺点 |
|------|------|------|
| **CMake (选定)** | HIP 官方支持, IDE 集成好, 跨平台 | 语法啰嗦 |
| Meson | 更简洁 | HIP 支持差 |
| Bazel | 大规模项目好 | 学习曲线陡 |
| Makefile | 简单 | HIP 集成需手写规则 |

**理由**: ROCm 官方提供 `FindHIP.cmake` 和各库的 CMake config (`hipblaslt-config.cmake`, `rccl-config.cmake`), CMake 是 HIP 生态的默认构建系统。本机 CMake 3.31 满足要求。

### 2.2 C++ 标准: C++20

| 选项 | 优点 | 缺点 |
|------|------|------|
| C++17 | 兼容性好 | 缺少 concepts, ranges |
| **C++20 (选定)** | concepts, ranges, coroutines, modules 预备 | 编译器支持要求高 |

**理由**:
- **Concepts**: 用于约束模板参数, 例如 `template<std::floating_point T> void add(...)` 比静态断言更清晰
- **Ranges**: 简化数据流水线, 例如 `data | std::views::take(n)`
- **std::format**: 替代 `std::stringstream`, 用于日志
- **std::span**: 零开销视图, 用于算子接口
- 本机 hipcc (AMD clang 23) 完整支持 C++20

### 2.3 HIP/ROCm 版本

本机环境:
- ROCm 7.13.0rc2
- HIP 7.13.99
- hipcc: AMD clang 23.0.0
- GPU: 4× AMD MI300X (gfx942)
- hipBLASLt: 1.3 (可用)
- RCCL: 1.0 (可用)

**版本选择**: 直接使用本机 ROCm 7.13。`ROCM_PATH=/opt/rocm` 已设为指向 `_rocm_sdk_devel` 的符号链接。

### 2.4 第三方依赖策略

| 类别 | 库 | 策略 |
|------|-----|------|
| 矩阵乘法 | hipBLASLt | **系统库** — ROCm 自带, 性能关键路径 |
| BLAS 基础 | hipBLAS / rocBLAS | **系统库** — 作为 hipBLASLt 的后备 |
| 多卡通信 | RCCL | **系统库** — ROCm 自带, N9 启用 |
| 测试 | Google Test | **FetchContent** — 自动下载, 无需手动安装 |
| 序列化 | 无 | **自己实现** — 朴素二进制 checkpoint 格式 |
| 分词 | 无 | **自己实现** — BPE 简化版 |
| 数据加载 | 无 | **自己实现** — 内存映射 + 预取 |
| 数学函数 | 自己实现 | RMSNorm, RoPE, SiLU, Softmax 等全部手写 |
| 内存分配 | 自己实现 | Storage 封装 hipMalloc/hipHostMalloc |
| 自动求导 | 自己实现 | 计算图 + 反向传播 (N5) |

**不用的库**: 不引入 Eigen, 不引入 fmt, 不引入 spdlog, 不引入 nlohmann/json。全部自己实现或用标准库。理由: 教学项目, 依赖越少越好。

### 2.5 测试框架: Google Test

| 选项 | 优点 | 缺点 |
|------|------|------|
| **Google Test (选定)** | 最流行, CMake 集成好, 参数化测试 | 体积大 |
| Catch2 | 单头文件, 语法简洁 | 参数化测试不如 GTest |
| doctest | 编译最快 | 社区较小 |
| 自定义 | 零依赖 | 维护成本高 |

**理由**: Google Test 的 `gtest_discover_tests()` 与 CMake 无缝集成, `EXPECT_NEAR` 对浮点数值验证非常方便, 参数化测试对算子测试 (不同形状/数据类型) 很有用。通过 FetchContent 自动获取, 不需要用户手动安装。

---

## 3. 目录结构设计及理由

```
microLLM-rocm/
├── CMakeLists.txt              # 顶层构建配置
├── .clang-format               # 代码格式化
├── .clang-tidy                 # 静态分析
├── .gitignore
├── README.md
│
├── src/                        # ====== 源代码 ======
│   ├── main.cpp                # 入口 (打印版本信息)
│   ├── core/                   # N0: Storage, Tensor, Device
│   │   ├── CMakeLists.txt
│   │   ├── storage.h           # Storage 接口定义
│   │   ├── tensor.h            # Tensor 接口定义
│   │   └── device.h            # (未来) Device, Stream, Event
│   │
│   ├── ops/                    # N1-N4: 算子实现
│   │   ├── CMakeLists.txt
│   │   ├── cpu/                # CPU 参考实现
│   │   │   ├── add.h/cpp       # 逐元素加法
│   │   │   ├── matmul.h/cpp    # 矩阵乘法
│   │   │   ├── embedding.h/cpp # 词嵌入查找
│   │   │   ├── rmsnorm.h/cpp   # RMS 归一化
│   │   │   ├── rope.h/cpp      # 旋转位置编码
│   │   │   ├── softmax.h/cpp   # Softmax
│   │   │   ├── swiglu.h/cpp    # SwiGLU 激活
│   │   │   └── cross_entropy.h/cpp # 交叉熵损失
│   │   └── hip/                # HIP 加速实现
│   │       ├── add.h/...       # HIP kernel 实现
│   │       ├── matmul.h/...    # 调用 hipBLASLt
│   │       └── ...
│   │
│   ├── autograd/               # N5: 自动求导
│   │   ├── CMakeLists.txt
│   │   ├── node.h              # 计算图节点
│   │   ├── graph.h             # 计算图
│   │   └── backward.h          # 反向传播引擎
│   │
│   ├── model/                  # N6: Transformer 组件
│   │   ├── CMakeLists.txt
│   │   ├── config.h            # ModelConfig 结构体
│   │   ├── embedding.h         # 词嵌入层
│   │   ├── attention.h         # 多头注意力
│   │   ├── ffn.h               # 前馈网络 (SwiGLU)
│   │   ├── transformer_block.h # Transformer Block
│   │   └── model_s.h           # Model-S 完整模型
│   │
│   ├── training/               # N7: 训练
│   │   ├── CMakeLists.txt
│   │   ├── optimizer.h         # SGD, AdamW
│   │   ├── trainer.h           # 训练循环
│   │   ├── checkpoint.h        # 权重保存/加载
│   │   └── data_loader.h       # 数据加载器
│   │
│   ├── inference/              # N8: 推理
│   │   ├── CMakeLists.txt
│   │   ├── kv_cache.h          # KV Cache
│   │   └── generator.h         # 自回归生成
│   │
│   └── multi_gpu/              # N9: 多卡
│       ├── CMakeLists.txt
│       ├── communicator.h      # RCCL 封装
│       └── distributed.h       # 分布式训练策略
│
├── tests/                      # ====== 测试 ======
│   ├── CMakeLists.txt
│   ├── tests_main.cpp          # 测试入口 + 冒烟测试
│   ├── core/                   # Storage/Tensor 测试
│   ├── ops/                    # 算子测试
│   ├── autograd/               # 自动求导测试
│   └── model/                  # 模型测试
│
├── notebooks/                  # ====== 教学文档 ======
│   ├── N0_storage_tensor.md    # N0: 指针 → Storage → Tensor
│   ├── N1_cpu_ops_basic.md     # N1: CPU 基础算子
│   ├── N2_cpu_ops_transformer.md  # N2: CPU Transformer 算子
│   ├── N3_hip_basics.md        # N3: HIP 编程基础
│   ├── N4_hip_ops.md           # N4: HIP 算子实现
│   ├── N5_autograd.md          # N5: 自动求导
│   ├── N6_transformer.md       # N6: Transformer 模型
│   ├── N7_training.md          # N7: 训练循环
│   ├── N8_inference.md         # N8: 推理与生成
│   └── N9_multi_gpu.md         # N9: 多卡训练
│
├── pa/                         # ====== 编程作业 ======
│   ├── PA0/                    # PA0: Storage 实现
│   ├── PA1/                    # PA1: CPU 算子
│   └── PA2/                    # PA2: Tensor 实现
│
├── data/                       # ====== 数据 ======
│   ├── README.md               # 数据集说明
│   └── .gitkeep
│
├── docs/                       # ====== 文档 ======
│   ├── INIT_RESEARCH.md        # 本文件
│   └── .gitkeep
│
├── cmake/                      # CMake 辅助模块
│   └── .gitkeep
│
└── scripts/                    # 辅助脚本
    ├── configure.sh            # 构建配置
    ├── build.sh                # 编译
    └── test.sh                 # 运行测试
```

**设计理由**:

1. **`src/core/` 最底层**: Storage 和 Tensor 不依赖任何其他模块, 是所有上层的基础
2. **`src/ops/` 分 CPU/HIP**: 明确的双路径设计。CPU 路径永远存在, 作为数值对照基准; HIP 路径在 GPU 可用时加速
3. **`src/autograd/` 独立模块**: 自动求导不是算子的一部分, 而是独立的反向传播引擎。这样可以让 N0-N4 不引入求导开销
4. **`notebooks/` 按课程编号**: N0-N9 对应课程主线, 每个文件是该阶段的教学文档
5. **`pa/` 编程作业**: 独立于 notebooks, 学生需要填空实现
6. **`tests/` 镜像 src 结构**: 每个模块有对应的测试目录

---

## 4. Model-S 配置 (15.8M 参数)

Model-S 是本课程的训练目标: 一个小到足以在单卡 MI300X 上快速训练, 又大到足以展示真实 LLM 行为的模型。

### 4.1 参数预算分析

LLaMA-style 架构, SwiGLU FFN, RMSNorm, RoPE, 无 bias。

参数计算公式:

```
总参数 ≈ 嵌入层 + N_layers × (注意力 + FFN + 层归一化) + 输出头

嵌入层:   V × D
每层注意力: 4 × D² (Q, K, V, O 各一个 D×D 投影)
每层 FFN:  3 × D × D_ff (gate, up, down 投影, SwiGLU)
每层 LN:   2 × D (RMSNorm weight, 每层 2 个: pre-attn + pre-ffn)
输出头:   V × D (与嵌入层共享时为 0)
```

### 4.2 选定配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `vocab_size` (V) | 8192 | 足够教学用 |
| `dim` (D) | 384 | 隐藏维度 |
| `n_layers` (L) | 6 | Transformer 层数 |
| `n_heads` (H) | 6 | 注意力头数 (head_dim = 64) |
| `n_kv_heads` | 6 | KV 头数 (MHA, 非 GQA) |
| `ffn_dim` (D_ff) | 832 | FFN 中间维度 (2.17× 隐藏维度) |
| `max_seq_len` | 512 | 最大序列长度 |
| `rope_base` | 10000 | RoPE 基频 |

### 4.3 参数量验证

```
嵌入层:     8,192 × 384      =  3,145,728
每层注意力: 4 × 384 × 384    =    589,824
每层 FFN:   3 × 384 × 832    =    958,464
每层 LN:    2 × 384           =        768
每层总计:                         1,549,056
6 层总计:                         9,294,336
输出头:     8,192 × 384      =  3,145,728
Final LN:                       384
----------------------------------------
总计:                          15,586,176
```

**15,586,176 参数（≈15.6M）**, 接近 15.8M 目标。

### 4.4 各层参数分布

| 组件 | 参数量 | 占比 |
|------|--------|------|
| 嵌入层 | 3,145,728 | 20.2% |
| 6× 注意力 | 3,538,944 | 22.7% |
| 6× FFN | 5,750,784 | 36.9% |
| 6× RMSNorm | 4,608 | 0.03% |
| Final RMSNorm | 384 | <0.01% |
| 输出头 | 3,145,728 | 20.2% |
| **总计** | **15,586,176** | **100%** |

### 4.5 训练资源估算

| 指标 | 值 |
|------|-----|
| 模型参数 | ~15.6M |
| FP32 显存 (模型) | ~60 MB |
| FP32 显存 (AdamW) | ~240 MB (2× 优化器状态) |
| FP32 显存 (梯度) | ~60 MB |
| 总训练显存 | ~360 MB + 激活值 |
| 单卡 MI300X 显存 | 192 GB |

结论: Model-S 在单卡 MI300X 上训练绰绰有余, 有大量余量做实验 (更大的 batch、更长的序列、混合精度等)。

---

## 5. 数据集候选和选择建议

### 5.1 候选数据集

| 数据集 | 规模 | 语言 | 适用阶段 | 理由 |
|--------|------|------|----------|------|
| **TinyShakespeare** | ~1MB | 英文 | N7 验证 | nanoGPT 经典选择, 训练极快, 可验证整个训练流程 |
| **WikiText-2** | ~17MB tokens | 英文 | N7-N8 | 语言模型标准 benchmark, 质量好 |
| **WikiText-103** | ~267MB tokens | 英文 | N7 扩展 | 更大规模验证 |
| **Tinystories** | ~2GB | 英文 | N8 | Karpathy 出品, 小模型也能生成有趣故事 |
| **OpenWebText** | ~38GB | 英文 | N9 | GPT-2 复现数据集, 接近真实预训练 |
| **中文维基百科** | ~2GB | 中文 | 进阶 | 如需中文能力 |

### 5.2 推荐方案

1. **N7 训练验证**: Tinystories (英文小故事, 生成有趣, 2GB 足够训练 15M 模型)
2. **N8 推理测试**: 用训练好的模型 + KV Cache 做自回归生成
3. **N9 多卡扩展**: OpenWebText (如果需要更大规模)

### 5.3 分词策略

由于我们使用 `vocab_size=8192` 的自定义小词表:

- **方案 A (推荐)**: 训练一个 BPE 分词器, 在 Tinystories 上拟合 8192 个 token
- **方案 B**: 使用 GPT-2 的 BPE 分词器, 截取前 8192 个 token (兼容性好)
- **方案 C**: 字节级分词 (byte-level), vocab_size=256, 极简但序列很长

推荐 **方案 A**: 自己训练 BPE, 完全自包含, 教学价值最高。

---

## 6. 下一阶段: N0 (Storage + Tensor) 详细实施计划

### 6.1 目标

实现 `Storage` 和 `Tensor` 的完整功能, 为后续所有模块奠定基础。

### 6.2 实施步骤

#### Step 1: Device 类 (0.5 天)

- [ ] 实现 `Device::to_string()`
- [ ] 实现 `Device` 的比较运算符
- [ ] 测试: CPU/HIP 设备创建和比较

#### Step 2: Storage CPU 实现 (1 天)

- [ ] 实现 CPU 内存分配: `new float[n]` / `delete[]`
- [ ] 实现构造函数、移动语义
- [ ] 实现 `fill()` — memset + 转换
- [ ] 实现 `copy_from()` — memcpy
- [ ] 实现 `to_vector()` / `from_vector()`
- [ ] 实现 `share()` — shared_ptr 共享
- [ ] 测试: 构造、填充、拷贝、共享、析构顺序

#### Step 3: Storage HIP 实现 (1 天)

- [ ] 实现 HIP 内存分配: `hipMalloc` / `hipFree`
- [ ] 实现 HIP `fill()` — hipMemset + kernel
- [ ] 实现 HIP `copy_from()` — hipMemcpy (支持 CPU↔HIP)
- [ ] 实现 HIP `to_vector()` — hipMemcpy D2H
- [ ] 测试: HIP 分配、D2D 拷贝、H2D/D2H 传输、设备间拷贝

#### Step 4: 形状和步长工具 (0.5 天)

- [ ] 实现 `compute_stride()` — 行主序步长
- [ ] 实现 `num_elements()` — 形状元素总数
- [ ] 实现 `is_valid_shape()` — 形状验证
- [ ] 测试: 各种形状的步长计算

#### Step 5: Tensor CPU 实现 (1.5 天)

- [ ] 实现从形状构造 Tensor
- [ ] 实现 `is_contiguous()` 判断
- [ ] 实现 `reshape()` / `unsqueeze()` / `squeeze()`
- [ ] 实现 `transpose()` — 步长变换
- [ ] 实现 `contiguous()` — 非连续时拷贝
- [ ] 实现 `to(Device)` — CPU→HIP 转移
- [ ] 实现 `to_string()` / `operator<<`
- [ ] 测试: 构造、形状变换、连续性判断、设备转移

#### Step 6: Tensor HIP 支持 (0.5 天)

- [ ] 验证 HIP Storage 的 Tensor 可以正常工作
- [ ] 测试: HIP Tensor 的形状变换

#### Step 7: 编程作业 PA0 (0.5 天)

- [ ] 设计 PA0: 学生实现 Storage 的 CPU 路径
- [ ] 编写 PA0 题目说明和测试用例

### 6.3 验收标准

- [ ] `Storage` 在 CPU 和 HIP 上都能正确分配和释放
- [ ] `Storage::share()` 引用计数正确
- [ ] `Storage::copy_from()` 跨设备传输正确
- [ ] `Tensor` 支持多维形状, 步长正确
- [ ] `Tensor::reshape()` 正确处理连续/非连续情况
- [ ] `Tensor::to(Device::HIP())` 正确转移设备
- [ ] 所有测试通过 (`ctest --output-on-failure`)

### 6.4 关键设计决策

1. **Storage 使用 `shared_ptr<Deleter>` 管理引用计数**: 比原始引用计数更安全, 支持 weak_ptr 观察
2. **Tensor 持有 `shared_ptr<Storage>`**: 天然支持零拷贝切片和视图
3. **offset_ 以元素为单位 (非字节)**: 与 PyTorch 一致, 简化索引计算
4. **CPU 默认使用 `new[]` 而非 `malloc`**: 对齐有保证, 且支持自定义 allocator
5. **先 CPU 后 HIP**: CPU 路径永远先实现, 作为数值对照

---

## 附录: 环境信息

```
OS:           Linux x86_64 (Debian)
GPU:          4× AMD Instinct MI300X VF (gfx942)
ROCm:         7.13.0rc2
HIP:          7.13.99
hipcc:        AMD clang 23.0.0
CMake:        3.31.10
g++:          13.3.0 (Ubuntu)
hipBLASLt:    1.3
RCCL:         1.0
ROCM_PATH:    /opt/rocm → /opt/python/lib/python3.13/site-packages/_rocm_sdk_devel
AMDGPU_ARCHS: gfx942
```
