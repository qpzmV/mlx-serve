# mlx-serve 中的 qwen4 架构实现详解（中文）

这个项目里，Qwen4 不是一个简单的“另一个模型名”，而是一个专门实现的架构分支，且它的结构和 qwen3_5 相关性很强，但又不是同一个东西。

关键点：

- 入口类型：`qwen4_exp`
- 相关文件：`src/qwen4_exp.zig`
- 逻辑上它是“Qwen3.5 trunk + Qwen4 新特性 + n-gram PLE + QSA + hyper-connections + MTP”
- 它是项目里非常关键的一条高级架构路线

---

## 一、Qwen4 在这个项目里的定位

在项目说明里，`qwen4_exp` 被描述为：

- Qwen3.8-Flash-Next
- 125B-A6B + 51B n-gram + 4B MTP
- qwen3_5 的 GDN + MoE trunk 被内嵌在 4 条 hyper-connection residual streams 中
- 还带有 n-gram PLE
- 还有 QSA sparse attention
- 最后还有 `hyper_connection_mixer`

它不是标准的 Qwen 3/3.5 结构，若按架构理解，应该看作：

> “Qwen3.5 语义主干 + Qwen4 的扩展控制流 + 一套 host-side n-gram 机制 + 专门的 MTP 架构”

---

## 二、关键文件和入口

最核心的实现文件是：

- [src/qwen4_exp.zig](../src/qwen4_exp.zig)

这个文件里最重要的几个主题：

- `hashing / n-gram table`
- `QSA mask`
- `hyper connection`
- `MTP`
- `forwardQwen4With`
- `forwardMoeBatchedDecode`
- `module-owned decode state`

它与其他文件的关系是：

- [src/transformer.zig](../src/transformer.zig)：统一 dispatch / arch 分发
- [src/mtp.zig](../src/mtp.zig)：独立 MTP 头实现
- [src/scheduler.zig](../src/scheduler.zig)：slot / spec / batch 调度
- [src/model_discovery.zig](../src/model_discovery.zig)：识别模型类型

---

## 三、核心架构特点：不是单纯“Qwen3.5 的升级版”

### 3.1 qwen3_5 GDN + MoE trunk

项目说明里明确说：

- trunk 仍然是 qwen3_5 的 GDN + MoE 结构
- 这是它的基础 backbone

也就是说，Qwen4 的“根”仍然接在 qwen3_5 的主干上，而不是完全另起炉灶。

### 3.2 hyper-connection residual streams

它最大的设计特征之一是：

- 4 个 hyper-connection residual streams
- 通过 `hcRead` / `hcWrite`
- 这些 residual stream 不是普通参数层的简单叠加
- 它们是一个并行的高阶状态通道

这个部分有两个重要性质：

- 它们参与了整个“hidden state”流动
- 它们和 QSA / n-gram / MTP 都在同一套生成环路里协同

### 3.3 n-gram PLE

Qwen4 还引入了 n-gram PLE（prefix-length extension / prediction-like mechanism）

它在项目里有非常明确的描述：

- host-side n-gram hash
- splitmix multipliers
- per-head primes
- eos-segment shifts
- `ngram_table.bin` 通过 mmap 读取

它并不是一个“临时优化”，而是模型中一套“外部硬编码候选路由”的机制。

### 3.4 QSA sparse attention

QSA 是 sparse attention 的一部分：

- 4-token blocks
- top-512 by relu-sum indexer score
- per-query tail
- lower-index ties 处理

它的核心特色是：

> 不是全 attention，而是按局部/索引选择有效 key/value 对。

这也是 qwen4 实现里最重要的性能/结构特征之一。

---

## 四、`forwardQwen4With`：真正的主 forward

这个函数是整套实现的核心入口。

它负责：

- 读取 hidden state
- 应用 vision embedding（如果有）
- 处理 hyper-connection stream
- 处理 QSA mask / sparse attention
- 处理 MoE forward
- 输出最后的 logits 或下一状态

### 4.1 它不是简单的单层卷积式前向

它本质上是一个复杂的多流融合前向：

- trunk/main stream
- hyper-connection streams
- sparse attention selection
- n-gram / PLE 辅助状态
- MTP 分支可能接在这条流上

### 4.2 这也是为什么它的状态管理很复杂

Qwen4 的 state 不只是某个 KVCache，而是包括：

- queue / cache / history / slot state
- PLE window
- QSA pooled block history
- aux state
- later MTP head support

这不是一个直接“单层 transformer”模型，而是一套多状态混合算子结构。

---

## 五、n-gram table 与 host-side lookup

这是 Qwen4 实现非常典型的一部分。

### 5.1 特点

- n-gram table 是一个 mmap 文件
- 不是模型权重里直接塞进内存的 tensor
- 需要通过 host-side gather 读取
- 走的是“只在需要时读取”的方式

### 5.2 为什么这个设计重要

因为它说明：

- 这个架构并不是纯粹依赖 MLX 层标准计算
- 它还混入了“外部统计索引 / 记忆表”来增强生成决策

这在项目的架构说明中是一个非常明确的特征。

---

## 六、QSA sparse attention：重点理解

QSA 是 qwen4 的一个高频关键点。

### 6.1 目标

把 attention 从“全量做”改成“按分块/评分选取关键范围”。

### 6.2 具体行为

项目里描述：

- 4-token blocks
- 每个 query 只看一部分 block
- 通过 relu-sum indexer score 全局筛选
- 位置上有 tail 和 tie handling

### 6.3 为什么这非常重要

因为它改变了：

- memory cost
- GPU 计算分配
- 生成过程中的 attention 范围
- prefill/verify 行为

这会直接影响：

- batch decode
- prefix cache
- spec-decode
- MTP 的可行性

---

## 七、MTP 与 qwen4 的关系

Qwen4 的 MTP 是非常关键的，因为它不是普通的一个小头，而是：

- 直接作用在 qwen4 的 pre-mixer / hidden stream 上
- 与 hyper-connection stream 耦合
- 可能依赖 QSA / PLE / MoE 相关结构

项目说明里指出：

- MTP 是 checkpoint 自己的 hyper-connected QSA + MoE layer
- 默认关闭 on MoE
- 通过 `--mtp` / `enable_mtp` 切开
- 它是“该模型自己带的一个 spec head”

### 7.1 这说明什么

说明 qwen4 的生成并不是单独一个 LLM forward，而是：

- trunk 主干做普通生成
- MTP head 参与 spec-decode 或多步验证
- 这会和 scheduler 的 `slotExclusiveDecode` 协同

### 7.2 它还与 module-owned decode state 相关

项目强调：

- qwen4 的 `Transformer.qwen4` 是只读 module state
- 其 hash + mmap 读状态是共享式
- slot 通过 `slotExclusiveDecode` 处理 MTP 独占场景

这说明它不是简单的单 slot 生成，而是要在更复杂的 runtime state 语义上协调。

---

## 八、`module_owned_state_fields` 与 `slotExclusiveDecode`

这是 Qwen4 很重要的一个设计思想。

### 8.1 共享只读 state

通过 `Transformer.qwen4` 这样的 read-only 模块状态，多个 slot 可以共享：

- hash
- mmap 表
- 统计结构

这减少了重复初始化成本。

### 8.2 但 MTP 仍需要 exclusive decode

因为 MTP head 上的状态不能无限共享，否则就会混合不同 slot 的历史。

最终项目里的设计是：

- read-only module state 可共享
- speculative generation head 或更高层状态仍然关心 slot exclusivity

也就是说：

> 共享不代表完全无状态，Qwen4 仍然会用 slot 的生命周期来控制更复杂状态。

---

## 九、`forwardMoeBatchedDecode` 与 batched decode

这个实现还与 batch decode 强耦合。

### 9.1 它不是单一 token 的串行前向

而是：

- 多个 slot 一起 decode
- GDN + PLE window 合并
- QSA mask 需要按 slot 进行 false padding

这意味：

- `slot` 的状态并不完全独立
- batched decode 会在某些条件下共享一部分算子路径
- 这和 scheduler 的 batching 逻辑必须保持一致

### 9.2 这是 Qwen4 高性能实现的一部分

它不仅要求数学正确，还要求在高并发条件下保持状态一致与资源可控。

---

## 十、与 qwen3_5 的关系：一条很强的架构继承链

Qwen4 最容易被误解的地方是：

- 它看起来像另一个模型对象
- 但它并不是完全独立的新架构

项目说明很明确：

- trunk 是 qwen3_5 的 GDN + MoE
- hyper-connections / QSA / n-gram PLE / MTP 是 qwen4 的扩展

所以真正更准确的理解是：

> qwen4 是在 qwen3_5 主干之上增量叠加了一组新机制，而不是从零重写一个 LLM。

这也解释了：

- 它和 qwen3_5 的很多实现接口很像
- 又在 forward / state / spec 方面有明显区别

---

## 十一、为什么它是项目里重要的架构分支

因为它同时具有：

1. 很强的算力优化逻辑
2. 复杂的 sparse attention
3. 多路 residual streams
4. n-gram / host-memory table
5. MTP spec head
6. batch decode 与 slot exclusivity 协调

这意味着它几乎把项目中最复杂的几个机制都集中在了一条架构上：

- hybrid/stateful forward
- speculative decoding
- optimized attention path
- Qwen 主干兼容 + 扩展增强

所以它是该项目中最能体现“架构工程”意味的分支之一。

---

## 十二、最适合的源码阅读入口

如果你想继续看源码，最推荐的入口顺序是：

1. [src/qwen4_exp.zig](../src/qwen4_exp.zig)
2. [src/transformer.zig](../src/transformer.zig)
3. [src/mtp.zig](../src/mtp.zig)
4. [src/scheduler.zig](../src/scheduler.zig)
5. [src/model_discovery.zig](../src/model_discovery.zig)

这样就能顺着看：

- 模型是如何识别为 qwen4_exp 的
- forward 是如何构造的
- MTP / QSA / PLE / hyper connection 如何协作
- scheduler 如何对其做 batched / exclusive decode

---

## 十三、结论

Qwen4 在这个项目中的实现，不是一个标准的“纯 decoder 增强”，而是一条高度工程化的高级架构路线：

- 继承 qwen3_5 主干
- 接入 hyper-connection streams
- 引入 QSA 稀疏注意力
- 融合 n-gram host-side table
- 走 MTP speculative head
- 强依赖 module-owned state 和 slot-exclusive decode

它充分体现了 mlx-serve 里“架构设计 + 运行时状态管理 + 性能优化”三者耦合的特点。

如果你继续想往下深读，下一步最合适的内容就是：

- MTP 具体实现
- 或者 QSA / n-gram 的 host-side 状态管理

这两部分会让你真正看到 qwen4 的“复杂性”在哪里。 
