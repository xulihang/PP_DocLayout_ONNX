# 模型量化与推理后端选型指南

> 面向 ONNX Runtime 部署的通用参考。结论来自实际测量（CPU / ORT 1.27），并尽量标注了适用范围。
> 核心观点：**"fp16 快不快"、"int8 能不能用"不取决于精度名字本身，而取决于模型结构、算子瓶颈和后端 kernel 支持。**

---

## 0. 结论先行（TL;DR）

| 场景 | 推荐 | 原因 |
|---|---|---|
| **GPU（CUDA / DirectML / CoreML / WebGPU）** | **fp16 优先** | 硬件原生支持、精度几乎无损、权重/显存减半，通常 1.5~3× |
| **CPU — 纯 CNN、大权重、batch=1** | **fp16 也可以很快（2~3×）** | 权重带宽减半 + CPU 侧 fp16 GEMM/Conv kernel |
| **CPU — Transformer / DETR / 混合精度图** | **fp32** | 缺 fp16 kernel、Cast 开销大，fp16 可能反而慢 1.5~2× |
| **int8 静态量化** | 仅纯 CNN + 充分标定 + 精度回归后使用 | CNN 常 2~4×；DETR/Transformer 容易精度崩塌 |
| **int8 动态量化** | Transformer 的保守选择 | 只压 MatMul/Gemm 权重，精度安全，但加速幅度有限（约 5~30%） |
| **WebGPU** | **只用 fp16** | WGSL 无 int8 计算路径，量化算子无 GPU 实现 |

---

## 1. 精度类型总览

| 精度 | 权重体积 | 典型加速 | 精度风险 | 适用性 |
|---|---|---|---|---|
| **fp32** | 1× | 基准 | 无 | 通用、保底方案 |
| **fp16** | 0.5× | GPU 1.5~3×；CPU 视模型而定 | 极低（接近无损） | GPU 默认首选；CPU 上纯 CNN 也能受益 |
| **int8 动态**（Dynamic Quant） | 0.5~0.9×（只压 MatMul/Gemm 权重） | CPU 约 5~30% | 低 | Transformer 安全折中 |
| **int8 静态**（QDQ / QOperator） | 约 0.25×（权重） | CNN 常 2~4× | **中~高**，依赖标定 | 纯 CNN + 有标定数据 |
| **int8（TensorRT / CUDA）** | 约 0.25× | GPU 2~4× | 中，需 PTQ 标定 | 服务端 GPU 批量推理 |
| **int4 / FP8** | 0.125~0.25× | 视硬件 | 高，需专门工具链 | LLM 等特定场景，本文不展开 |

补充说明：

- **量化节省的主要是权重，不一定是激活/工作区。** int8 静态模型在运行时仍可能把激活反量化回 fp32（尤其 transformer 中不被量化的算子），峰值内存不一定按 1/4 下降。
- **fp16 在 CPU 上的内存表现不可预测**：权重文件减半，但 ORT 可能为计算准备 fp32 副本/prepack，首跑峰值反而更高（本次 OCR 实测：fp32 256MB vs fp16 298MB）。

---

## 2. 体积 / 速度 / 内存的本质

### 2.1 体积

- 理论权重字节数 = 参数量 × 每参数字节：fp32 4B、fp16 2B、int8 1B。
- 实际文件还要加图结构、量化参数（scale/zero-point）、对齐开销。
- 小模型里图结构占比高，量化收益会打折扣；大模型基本按理论比例。
- 导出时可能生成 **external data**（`.onnx` + `.onnx.data`），权重在 data 文件里，两个文件必须同目录，可合并为单文件（总量 < 2GB 时）。

### 2.2 速度

决定 fp16/int8 是否加速的三个因素：

1. **后端是否有对应 kernel**（CPU 的 fp16 支持弱；int8 在 x86 AVX2/VNNI 上成熟）。
2. **模型是带宽瓶颈还是计算瓶颈**：batch=1 的 CNN/大 GEMM 通常是**内存带宽瓶颈**，权重减半就能接近 2×；计算密集型 + 大 batch 则要靠 Tensor Core/DP4A。
3. **图结构**：混合精度图（大量 Cast）、缺 kernel 的算子（LayerNorm、Softmax、GridSample 等）会吃掉收益。

### 2.3 内存

峰值内存 ≈ 运行时基线 + 权重 + 激活/工作区 + 框架缓冲。

- **OCR 这类模型**：权重主导（165MB 权重占 256MB 峰值的大部分），量化直接见效。
- **DETR/Transformer 检测模型**：工作区主导（本次 800×800 检测模型峰值 966MB，权重仅 142MB），只压权重对内存帮助有限。
- **GPU**：fp16/int8 的权重和中间 buffer 直接按比例缩小，收益明确；**CPU fp16** 可能因转换/prepack 抵消甚至反超。

---

## 3. 实测案例（同一台机器）

> 环境：Intel Comet Lake 12 线程、ORT 1.27.0、CPU EP、Python。
> 测法：初始化 3 次取中位数；推理连续 5 次；峰值内存为进程 `peak_wset`（含约 54MB 运行时基线），首次推理后采样。

### 3.1 检测模型（Transformer/DETR 类，PP-DocLayoutV3，800×800）

| 版本 | 体积 | 初始化 | 单次推理 median | 峰值内存 | 精度 |
|---|---|---|---|---|---|
| fp32 | 141.9 MB（.onnx 5.1 + .data 136.8） | ~0.90 s（首次冷启动 2.2 s） | **1.23 s** | **966 MB** | 基准 |
| fp16 | 74.3 MB（单文件） | 1.47 s | 2.03 s | 621 MB | 与 fp32 检测结果一致 |
| int8 静态（QDQ，Conv+MatMul） | 44.2 MB | — | — | — | **崩溃**：最高置信度 0.089（fp32 0.968），0 个框 |
| int8 静态（仅 Conv） | 65.9 MB | — | — | — | 同样崩溃 |
| int8 动态 | 119.8 MB | — | ~1.1 s（约快 5%） | — | 基本可用 |

结论：**该模型在 CPU 上 fp32 最优；fp16 更省内存但慢 1.65×；int8 静态不可用。**

### 3.2 OCR 模型（纯 CNN + 大分类头，OCR CTC）

| 版本 | 体积 | 初始化 | 48×260 推理 | 48×800 推理 | 峰值内存 | 精度 |
|---|---|---|---|---|---|---|
| fp32 | 165.0 MB | 586 ms | 312 ms | 948 ms | 256 MB | 基准 |
| **fp16** | **82.6 MB** | **329 ms** | **153 ms** | **329 ms** | 298 MB | argmax 100% 一致、解码文本相同 |

结论：**该模型 fp16 全面胜出：快 2.0~2.9×、体积减半、精度无损。**

### 3.3 为什么同样是 fp16，一个更快一个更慢？

| 因素 | 检测模型 | OCR 模型 |
|---|---|---|
| 图结构 | 混合精度：upsample/trilinear 等留在 fp32，Cast 多 | 纯 fp16 CNN，无混合 Cast |
| 主要算子 | Conv + LayerNorm/Softmax/GridSample（CPU 缺 fp16 kernel） | Conv/BN/ReLU + 大 GEMM（有 fp16 kernel 路径） |
| 瓶颈 | 激活/工作区，不是权重带宽 | batch=1 下的大权重读取，带宽瓶颈 |
| fp16 结果 | 慢 1.65× | 快 2.0~2.9× |

---

## 4. int8 的适用边界

### 4.1 动态 vs 静态

- **动态量化**：只把 MatMul/Gemm 的权重转 int8，激活在运行时动态算 scale。改动小、精度安全；对 CNN 帮助很小（Conv 权重没压）。
- **静态量化（QDQ）**：权重 int8 + 激活 int8，需要**标定数据集**；CNN 上收益最大，但对异常值和模型结构敏感。

### 4.2 静态量化常见坑

1. **离群激活撑大量化步长**。本次检测模型某层实际范围 [-17, 15] 但标准差仅 0.42，MinMax 得到的步长 0.35，几乎抹平信息。→ 用 **Percentile/Entropy** 标定，或在标定前裁剪离群值。
2. **误差逐层累积**。实测每层相对误差从 0.6% 涨到 7.5%、14%，深层到 31%，DETR 解码器直接失效。→ 逐层比对中间张量，排除敏感层（`nodes_to_exclude`）。
3. **Transformer 的注意力/分类头对量化极敏感**。→ 优先只量化 Conv；MatMul 走动态量化或保留 fp32。
4. **激活量化不等于内存下降**。大量算子未量化，边界反复 Quantize/Dequantize，速度可能不升反降。
5. **必须做精度回归**。不要只看 loss/单层误差，要跑业务后处理（NMS、CTC 解码、阅读顺序等）后的最终结果。

### 4.3 int8 使用清单

- [ ] 模型是 CNN 或有充分证据表明对量化不敏感
- [ ] 有覆盖真实场景的标定集（几十~几百张）
- [ ] 使用 QDQ、per-channel 权重、Percentile/Entropy 标定
- [ ] 对中间层做逐层误差分析，必要时排除敏感层
- [ ] 最终任务指标（检测框/文本/顺序）回归通过
- [ ] 实测速度与内存，而不是只看模型体积

---

## 5. 后端选型

| 后端 | 原生精度支持 | 优势 | 限制 | 典型场景 |
|---|---|---|---|---|
| **CPU** | fp32、int8（x86 AVX2/VNNI，ARM dotprod）；fp16 视算子/kernel | 零依赖、最通用；int8 生态成熟 | 吞吐有限；fp16 对 Transformer 常无收益 | 桌面应用、嵌入式、无 GPU 环境 |
| **CUDA（含 TensorRT）** | fp32/fp16/int8 | 吞吐最高；fp16 几乎无损，int8 走 Tensor Core | 需 NVIDIA GPU + CUDA/cuDNN；int8 需 TensorRT 标定 | 服务端批量、GPU 工作站 |
| **DirectML（Windows）** | fp32/fp16（部分 int8） | 消费级 Windows GPU 通用加速 | 驱动/机型差异，需实测 | Windows 桌面应用（Java 可用） |
| **CoreML（macOS）** | fp16/int8 | Apple Silicon / Neural Engine | 仅 macOS | Mac 桌面应用 |
| **WebGPU（浏览器）** | **f32 / f16** | 浏览器内 GPU、零安装、跨平台 | **无 int8 计算**；shape 限制多；显存有上限 | 网页版 OCR/文档处理 |
| **WASM（浏览器）** | fp32、int8（SIMD） | 兼容性最好 | 速度有限，无 GPU | 无 WebGPU 时的网页兜底 |
| **WebNN** | fp32/fp16/int8（逐步支持） | 浏览器原生 ML API | 可用性和算子覆盖仍在发展 | 浏览器高性能推理的未来方向 |

选择顺序建议：

1. 有 NVIDIA GPU 且可部署 → **CUDA/TensorRT，fp16 起步，必要时 int8**
2. Windows/macOS 桌面 → **DirectML/CoreML，fp16**
3. 浏览器 → **WebGPU + fp16**；不支持时回退 **WASM + fp32/int8**
4. 纯 CPU → **CNN 可试 fp16；Transformer 用 fp32；int8 只在标定充分时**

---

## 6. 为什么 WebGPU 用 fp16 而不是 int8

1. **WGSL 没有可用的 int8 计算路径**。着色器类型只有 `f32`、`f16`（`shader-f16` 特性）、`i32/u32`；没有 int8 向量类型，也没有 DP4A / 点积内建函数。即使把 int8 打包进 u32，解包和逐元素乘加要消耗大量 ALU 和寄存器，抵消收益。
2. **量化算子没有 GPU 实现**。ONNX Runtime Web 的 WebGPU EP 用 WGSL 手写 kernel，只覆盖浮点算子（Conv/MatMul 等）。`QLinearConv`、`QLinearMatMul`、`MatMulInteger` 没有 WebGPU 版本——int8 模型要么回退到 WASM，要么在加载时反量化，**GPU 加速拿不到**。ORT Web 的官方建议就是：WebGPU 用 fp16，WASM 用 int8/QDQ。
3. **fp16 是硬件原生能力**。主流 GPU（NVIDIA/AMD/Intel/Apple）都支持 f16 计算与 f16 buffer，Chrome/Edge 的 WebGPU 也广泛支持 `shader-f16`；fp16 权重和显存缓冲直接减半，缓解浏览器的显存/内存限制。
4. **int8 的加速依赖厂商私有能力**。DP4A、INT8 Tensor Core、per-channel 整数卷积都是硬件/驱动特性，WebGPU 标准没有暴露，无法可移植地使用。
5. **精度上 fp16 足够**。实测 OCR fp16 的 argmax 100% 一致、文本完全相同；检测模型 fp16 与 fp32 结果一致。int8 反而有标定和精度崩塌的风险。

> 例外：未来 WebNN（`navigator.ml`）或厂商 WebGPU 扩展可能提供 int8 加速，届时可重新评估。

---

## 7. 落地决策清单

1. **先确定后端**（CPU / CUDA / DirectML / CoreML / WebGPU / WASM）。
2. **优先试 fp16**（GPU 必试；CPU 上纯 CNN 也值得试）：比较体积、初始化、推理、峰值内存、精度。
3. **CPU + Transformer/DETR**：如果 fp16 变慢，回到 fp32；再考虑 int8 动态量化作为折中。
4. **考虑 int8 静态**：仅在纯 CNN、有标定集、精度回归通过时采用；用 Percentile/Entropy、per-channel，必要时排除敏感层。
5. **以终态指标验收**：不要只看模型输出，要跑完整后处理（NMS、CTC 解码、阅读顺序等）。
6. **实测四个维度**：体积、初始化时间、单次推理时间、峰值内存；记录测试机型和线程数。
7. **工程细节**：
   - 按**输出名字**取值，不要依赖输出顺序（不同导出工具顺序可能不同）；
   - `.onnx` + `.onnx.data` 必须同目录，或合并成单文件；
   - 注意输入 shape 是否固定（部分导出是 `[1,3,800,800]`，部分是符号化 batch）；
   - 某些 fp16 导出在 ORT 高优化等级会触发融合 bug（本次遇到 LayerNorm 融合失败），可降到 `EXTENDED` 或升级 ORT。

---

## 8. 附录：测量方法与注意事项

### 8.1 测速脚本要点（Python / ORT）

```python
import onnxruntime as ort, numpy as np, time, statistics

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 或 EXTENDED
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# 初始化：重复 3 次取中位数
inits = []
for _ in range(3):
    t = time.perf_counter()
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    inits.append(time.perf_counter() - t)

# 推理：连续 5 次，分别记录
runs = []
for _ in range(5):
    t = time.perf_counter()
    sess.run(None, feeds)
    runs.append(time.perf_counter() - t)

print("init median:", statistics.median(inits))
print("infer mean/median/min:", statistics.mean(runs), statistics.median(runs), min(runs))
```

### 8.2 容易造成误测的点

- **首次推理**包含内存池/arena 分配，通常比后续慢（本次 fp32 检测模型首次 1.4s，后续 1.1~1.2s）。报告时区分首跑和稳定值。
- **冷启动读盘**会让第一次初始化明显偏慢；初始化测 3 次更可靠。
- **线程数**直接影响 CPU 结果，记录 `intra_op_num_threads`/物理核心数。
- **输入形状**决定计算量：OCR 用 48×260 和 48×800 结果差 6 倍，必须用业务真实尺寸。
- 不同 ORT 版本的优化器/kernel 差异很大，跨版本比较要注明版本号。
- 峰值内存包含 Python/框架基线，建议同时给出"减去基线"的增量。

### 8.3 本次测量环境

- CPU：Intel Comet Lake，12 逻辑线程
- ONNX Runtime：1.27.0（Python），CPU Execution Provider
- 模型：PP-DocLayoutV3（检测/Transformer）、OCR CTC（CNN）
- 所有时间单位：毫秒（ms）；体积单位：MB（10^6 字节）
