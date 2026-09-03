# Qwen4 的 QSA 与 n-gram host-side state 机制详解（中文）

这一节继续围绕 Qwen4 最关键的两大实现特征展开：

- QSA sparse attention
- n-gram host-side state / index table

这两者共同构成了 Qwen4 的“高性能 + 高状态复杂度”特征，几乎是它区别于 qwen3_5 的核心实现。

关键文件：

- [src/qwen4_exp.zig](../src/qwen4_exp.zig)
- [src/transformer.zig](../src/transformer.zig)
- [src/mtp.zig](../src/mtp.zig)
- [src/scheduler.zig](../src/scheduler.zig)

---

## 一、Qwen4 的两条最关键支线

Qwen4 不是只有一个主 trunk。它的关键增强主要有两条：

1. QSA sparse attention
2. n-gram / PLE host-side state

这两条设计共同决定了：

- 计算成本如何降低
- attention 选择机制如何细化
- 生成时如何利用历史上下文
- 为什么它需要更复杂的 slot / cache / rollback 状态

---

## 二、QSA：稀疏注意力的核心

QSA 的中文可以理解为：

- Query-Selective Attention
- 或更具体地：按选择性块去做 sparse attention

### 2.1 典型实现特征

项目说明中给出的几个关键点：

- 4-token block
- top-512 by relu-sum indexer score
- per-query tail
- lower-index ties 处理
- mask 由 query / key 的有效位置决定

这说明它不是简单的 local attention，而是：

> 每个 quer y 只关注一组“值得看的” blocks，而不是整个历史全量注意力。

### 2.2 为什么是“QSA”而不是普通 sparse attention

它的重点不只是减少计算量，而是：

- 需要用一个 score/indexer 去筛选哪些位置是最有价值的
- 这些位置会随 query 不同而变化
- 它直接影响后续的 MTP / decode / cache 行为

所以它在 Qwen4 里不是一个辅助优化，而是架构主干的一部分。

---

## 三、QSA 如何和 forward 交互

Qwen4 的典型 forward 流程中，QSA 不是“扔在最后做一个优化”，而是：

- 在 attention 侧参与 mask 计算
- 在 decode / prefill 中筛选 query-key 关系
- 与 hyper-connection stream / hidden state 一起工作
- 影响下游的 MTP 和缓存状态

### 3.1 它改变的是“哪个历史位置被看见”

注意：

> QSA 不是改变模型的表示维度，而是改变“有效 attention 范围”。

这直接决定了：

- 哪些 key/value 被纳入计算
- 某个 token 的有效上下文历史
- 生成时模型的“注意力视野”边界

### 3.2 这也解释了为什么它对性能很重要

因为 attention 常常是生成中最重的一部分，QSA 能显著降低：

- KV 读取成本
- attention mask 构造成本
- 复杂历史的计算量

---

## 四、n-gram / PLE：host-side 记忆结构

Qwen4 的第二大关键是：

- n-gram hash
- host-side table
- PLE / ngram 感知路径

### 4.1 这不是标准模型权重的一部分

项目说明很明确：

- n-gram table 是 host-side mmap 的表
- 使用 splitmix multipliers / per-head primes / eos shift
- 通过 `ngram_table.bin` 做 gather

这非常像一种“外部索引结构”，它不完全依赖模型自己的参数，而是：

> 在 runtime 中利用 host 侧的历史特征表来辅助 token 选择。

### 4.2 为什么它很关键

因为它靠“历史/模式匹配”来加强生成，不完全依赖纯 transformer hidden state。

这带来两个结果：

- 更高的生成稳定性
- 更强的使用历史模式的能力

同时也意味着：

- runtime 更复杂
- 状态管理更强
- prefix cache / rollback / MTP / slot 管理必须一起考虑

---

## 五、n-gram table 与 cache / slot 的关系

### 5.1 它不是一个单纯静态查表

它依赖：

- 当前 token history
- 当前 qwen4 slot 的历史状态
- 当前的 prompt / decode 位置
- 可能的 eos / boundary 状态

所以它不是“一个全局常量”，而是：

> 依赖运行时上下文的动态辅助状态。

### 5.2 这也是 why Qwen4 复杂度很高

Qwen4 的生成路径常常需要同时维持：

- token history
- attention mask
- sparse selected blocks
- n-gram 判断 table
- MTP speculative head
- prefix cache snapshot

这些状态之间不是完全独立的，而是相互影响的。

---

## 六、QSA 与 n-gram 共同影响 MTP

这两个机制从设计上并不是彼此独立，而是相互影响的。

### 6.1 QSA 决定“哪部分历史有效”

### 6.2 n-gram 决定“历史中哪些模式值得利用”

### 6.3 MTP 又基于这些状态做预测

所以构成了一个闭环：

```text
历史上下文
   ↓
QSA 选取有效 block
   ↓
ngram / PLE 提供历史模式辅助
   ↓
MTP 在这些状态上做候选预测
   ↓
scheduler 进行 accept / rollback / continue
```

这套闭环是 Qwen4 的核心实现价值所在。

---

## 七、为什么这些状态必须和 scheduler 协同

Qwen4 的复杂性不只是模型里的参数多，而是它在 runtime 时需要：

- 维护多个历史视角
- 处理 per-query sparse selection
- 共享/隔离 slot state
- 处理 speculative verify 的 partial accept
- 更新 cache / offset / rollback

### 7.1 一旦缺失某一层状态，生成会出现灾难性问题

比如：

- 错误的 QSA mask 会让历史被看错
- 错误的 n-gram lookup 会让生成偏向错误模式
- 错误的 MTP checkpoint 会让 spec-decode 接受错误 token
- 错误的 slot isolation 会把多个请求混合状态

所以在 Qwen4 里，架构和 scheduler 的关系比普通模型更加紧密。

---

## 八、为什么它不是简单地“把 attention 改成 sparse”

这是最容易误解的一点。

Qwen4 不是单纯做一个 sparse attention 变体，它还叠了：

- hyper-connection residuals
- n-gram host index
- MTP head
- batch decode transport
- slot-exclusive decode semantics

也就是说：

> 它是一个完整的“结构 + runtime state + 生成策略”组合体。

---

## 九、这两大机制对性能的影响

### 9.1 QSA

它的主要价值是：

- 减少不必要的 attention 成本
- 让每一 query 尽量只看关键历史位置
- 提高长上下文下的效率

### 9.2 n-gram / PLE

它的主要价值是：

- 让模型更依赖历史模式和 token 统计特征
- 提高生成一致性
- 让 MTP / spec head 更有判断依据

两者结合之后，Qwen4 就表现出：

- 更强的长上下文性能
- 更复杂的状态管理
- 更高的工程复杂度

---

## 十、最关键的源码位置

如果你继续读源码，最值得看的地方是：

1. [src/qwen4_exp.zig](../src/qwen4_exp.zig)
2. [src/transformer.zig](../src/transformer.zig)
3. [src/mtp.zig](../src/mtp.zig)
4. [src/scheduler.zig](../src/scheduler.zig)

重点看这些逻辑：

- `forwardQwen4With`
- `qsaMask`
- `ngram_table.bin` 的读取与 gather
- `slotExclusiveDecode`
- `MtpCacheRef`
- `ssmRollbackFromCapture`

---

## 十一、结论

Qwen4 的 QSA 和 n-gram host-side state，是它与 qwen3_5 的本质差别之一。

它们共同构成：

- 稀疏但有选择的 attention
- 历史模式驱动的生成辅助
- 更复杂的 MTP / speculative decode
- 更强的 runtime state 管理需求

因此，Qwen4 在这个项目中不是“另一个模型”，而是：

> 一个典型的、高度状态驱动的高级生成架构实现。

如果继续往下深入，下一步最自然的内容就是：

- Qwen4 的 hyper-connection streams 与 residual state
- 或者 Qwen4 与 qwen3_5 的完整架构对照。
