# Qwen4 最终架构总结（中文）

这篇文档把前面 Qwen4 相关内容收束成一个最终总结，给出一份最适合你直接记忆和后续源码阅读的“全景图”。

目标不是重复细节，而是把 Qwen4 作为一个架构对象完整地描述出来。

---

## 一、Qwen4 在 mlx-serve 中的定位

Qwen4 在这个项目里并不是一个普通的模型名，而是一条高度工程化的架构分支。

它的类型是：

- `qwen4_exp`

它和普通 Qwen 结构的关系是：

- 继承 qwen3_5 的主干思路
- 但在 trunk 之外加入了更复杂的状态流与 spec 机制
- 重点不是单纯“更大更强”，而是“更复杂的运行时结构”

---

## 二、它的核心主干：仍然是 qwen3_5 的 GDN + MoE trunk

最关键的一点：

- Qwen4 的根基仍然是 qwen3_5 的 GDN + MoE trunk

也就是说：

- 它不是彻底换了一套基础模型架构
- 它是在已有主干之上叠加新的高阶机制

这决定了：

- 它和 qwen3_5 的接口与骨架有很强联系
- 但它的生成和状态管理已经不是同一层复杂度

---

## 三、Qwen4 的四大关键结构

Qwen4 的核心可以概括成四类机制：

### 1. Hyper-connection residual streams

特点：

- 4 条 parallel residual streams
- 通过 `hcRead` / `hcWrite` 读写
- 最后由 `hyper_connection_mixer` 汇总

它的作用：

- 让模型不只是单纯的主干 forward
- 额外维护跨层状态流
- 在最终输出前把这些状态重新混合进入主干语义

一句话：

> 它让模型拥有额外的跨层记忆通道。

---

### 2. QSA sparse attention

特点：

- 使用 block-level sparse attention
- 通过 indexer score 只保留关键位置
- 每个 query 只看部分 key/value blocks

它的作用：

- 减少 attention 计算量
- 让历史上下文更有选择性地参与生成
- 提高长上下文下的效率

一句话：

> 它让 attention 从“全量看历史”变成“只看重要历史”。

---

### 3. n-gram / PLE host-side state

特点：

- host-side table
- `ngram_table.bin` 读取
- 利用 splitmix / per-head primes / eos shifts 等 hash-based 信息

它的作用：

- 把历史模式统计整合进生成决策
- 让生成不只依赖纯 hidden state
- 在 runtime 中用外部索引结构辅助推断

一句话：

> 它让模型带上了历史模式索引辅助。

---

### 4. MTP speculative head

特点：

- 模型自带 MTP head
- 不是单纯最终 logits head
- 与 hyper stream + QSA + MoE 协同

它的作用：

- 让生成过程进入 spec-decode
- 提前预测未来 token
- 通过 verify / accept / rollback 控制生成推进

一句话：

> 它让生成过程更像“多 token speculative 预测 + 验证”。

---

## 四、Qwen4 的真正本质：不是模型增强，而是运行时结构增强

Qwen4 最容易被误解的地方是：

- 它看起来像是一个更高级的模型
- 但它真正的关键不是“参数更大”，而是“结构和运行时状态更复杂”

它融合了：

- 主干模型
- sparse attention
- hyper-flow residual
- history index
- speculative head
- slot state / batch decode / accept-rollback

所以它是一个典型的：

> 高度状态驱动的生成架构，不只是一个单纯 transformer 结构。

---

## 五、Qwen4 的主前向链路：`forwardQwen4With`

最核心的入口是：

- `forwardQwen4With`

它负责：

- 读取 hidden state
- 应用 vision embedding（如果有）
- 执行 hyper-connection 流读写
- 应用 qsa mask / sparse attention
- 参与 MoE trunk
- 最终输出主干状态或 MTP 相关输出

这条链路的关键意义在于：

> 不是一个纯线性 forward，而是多流、多状态、多阶段的协同前向。

---

## 六、Qwen4 与 MTP 的关系

Qwen4 的 MTP 并不是“一个普通小头”，而是：

- 与 hyper stream 相关
- 与 QSA / MoE / hidden state 有强耦合
- 与 scheduler 的 slot-exclusive decode 配合

也就是说：

- 它不能只看作一个单独输出头
- 它必须看作运行时中的一条 spec-decode 分支

---

## 七、Qwen4 与 scheduler 的深度绑定

在这个项目里，Qwen4 的实现不可能完全独立运行。

它依赖：

- slot lifecycle
- batched decode
- partial accept / rollback
- cache 更新与恢复
- `slotExclusiveDecode`

这说明：

> Qwen4 的正确性不仅取决于参数和 forward 算子，也取决于调度器是否在正确的状态下推进它。

---

## 八、简短记忆版

你可以把 Qwen4 简单记成：

```text
Qwen4 = qwen3_5 trunk + hyper residual streams + QSA sparse attention + n-gram host state + MTP spec head
```

而它最关键的实现思想是：

```text
让模型在主干之外多维护一套跨层状态流，并在生成时利用 sparse attention + n-gram + spec head 协同推进。
```

---

## 九、最值得继续读的源码入口

你后面直接继续看源码，最有效的入口是：

1. [src/qwen4_exp.zig](../src/qwen4_exp.zig)
2. [src/transformer.zig](../src/transformer.zig)
3. [src/mtp.zig](../src/mtp.zig)
4. [src/scheduler.zig](../src/scheduler.zig)

最值得关注的几个标志：

- `forwardQwen4With`
- `hcRead` / `hcWrite`
- `hyper_connection_mixer`
- `qsaMask`
- `ngram_table.bin`
- `slotExclusiveDecode`
- `MtpCacheRef`

---

## 十、最终结论

Qwen4 在 mlx-serve 中的本质是：

- 一条建立在 qwen3_5 trunk 之上的高复杂度生成架构
- 以 hyper-connection、QSA、n-gram、MTP 为核心
- 强依赖运行时状态管理与 scheduler 协同
- 不只是一个更大的模型，而是一套更复杂的生成执行系统

如果你已经读完前面的几篇 Qwen4 文档，那么现在对它的理解，应该已经从“模型名”提升到了“架构实现与运行时状态协同”的层次。
