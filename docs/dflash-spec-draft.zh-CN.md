# DFlash 与 spec draft 的实现机制（中文）

这一节继续沿着项目的性能增强主线往下走，重点解释：

- DFlash 是什么
- 为什么它属于 spec draft
- 它和 scheduler / generate / transformer 的关系
- 它为什么比普通 greedy decode 更复杂
- 它在 Qwen / hybrid / MTP 等场景中如何协同

关键文件：

- [src/dflash.zig](../src/dflash.zig)
- [src/drafter.zig](../src/drafter.zig)
- [src/generate.zig](../src/generate.zig)
- [src/scheduler.zig](../src/scheduler.zig)
- [src/transformer.zig](../src/transformer.zig)

---

## 一、什么是 spec draft

Spec draft（speculative draft）可以理解为：

- 不是每个 token 都等大模型真正算一遍
- 先让一个更轻量的 draft 路径预测一段候选 token
- 大模型再确认这段 token 是否应该接受

它的核心思想是：

> 用“预测 + 验证”的方式减少大模型的真实计算量，从而提升吞吐。

这个项目中，DFlash、Drafter、MTP、PLD 等都属于这类加速机制的不同实现方式。

---

## 二、DFlash 的定位

DFlash 在项目里被明确看作：

- 一种 block-drafter
- 它的 key 是 `block_size`
- 它有自己的 config contract
- 它会从一个 assistant forward 里生成 block-size-1 的 draft
- 然后通过 verify 路径确认接受数量

也就是说：

> DFlash 本质上不是“默认 decode”，而是一条 block-wise speculative generation 路径。

---

## 三、为什么它叫 DFlash

名字本身就说明它偏“draft + flash”式的快速 prefetch / speculation 逻辑。

它的重点不是长文本生成，而是：

- 先用小或特殊的前向辅助逻辑推测若干 token
- 再接着做更严格的主模型 verify

因此它和普通 token-by-token decode 的差别非常大。

---

## 四、DFlash 的关键机制：block draft

DFlash 不是“一个 token 一个 token guess”，而是：

- 以 block 为单位
- 先利用一组 assistant forward 生成 draft block
- block 的大小和配置中的 `block_size` 相关
- 一次 draft 会产生 `block_size - 1` 个候选 token（按项目描述）

### 4.1 这是 spec 的本质

它不是在最终输出前直接产生最终 token，而是在：

- 先生成一段候选内容
- 再交给 verify 路径确认

所以它真正的结构是：

```text
draft block -> verify -> accept subset -> continue
```

---

## 五、DFlash 的 config contract

项目里面反复强调：

- DFlash 是按 config contract key 识别的
- 不是随便开一个 flag 就能使能
- 关键字段包括：
  - `block_size`
  - `mask_token_id`
  - `target_layer_ids`
  - `dflash_config`

### 5.1 为什么 contract 很重要

因为 DFlash 的激活并不是只靠一个 `--drafter` 这么简单。

它必须符合：

- 配置文件中的 contract
- 模型自己是否内置了 DFlash 状态
- runtime 是否支持这条路径

如果配置不匹配，那么它就不能作为真正的 draft path 运行。

---

## 六、它和 scheduler 的关系

DFlash 不能脱离 scheduler 单独存在，因为它涉及：

- 当前 slot 是否允许 draft
- 当前 slot 处于哪个 generation phase
- 是否需要走 accept / reject / rollback
- draft 的 cache 是否需要被 commit 或回滚

### 6.1 scheduler 维护的是“生成状态”

它不是负责算内部 token，而是负责：

- slot 的状态切换
- draft 是否启用
- draft 块是否被接受
- 某轮 draft 的 cache / context 状态应该如何推进

### 6.2 DFlash 是一条“状态驱动的 spec 分支”

因此在逻辑上，它属于 scheduler 控制下的生成加速机制，而不是普通模型前向逻辑。

---

## 七、DFlash 与 transformer 的关系

DFlash 最终还是要落到具体 forward 和 cache 机制。

它使用：

- assistant forward
- capture_layers
- block K/V cache
- verify 流程

### 7.1 它不直接与那些复杂的主干逻辑完全叠加

而是：

- 使用一条专门的 draft 侧链
- 从 trunk 里拿到必要的 hidden / cache
- 然后形成一个 block draft

### 7.2 这也是为什么它被放在 `dflash.zig` 而不是普通 `transformer.zig`

因为它实质上是：

> 一套“专门的 speculative generation runtime”，而不只是一个常规的层实现。

---

## 八、DFlash 的 verify / accept 流程

DFlash 的核心并不是“预测”，而是“验证”。

### 8.1 预测阶段

使用 assistant forward / draft path，生成一批候选 token。

### 8.2 验证阶段

主 trunk 对候选序列进行验证，判断：

- 这一段是不是合理
- 多少 token 应该被接受
- 哪部分应该被回退

### 8.3  partial accept

项目里对 partial accept 非常重视，因为它确保：

- 不需要一次性全部接受
- 只接受可验证的前缀
- 其余部分回滚到正确状态

这是 spec draft 里非常关键的一套状态保证。 

---

## 九、为什么 block 级别更复杂

如果仅仅是一个单 token 的 draft，那实现会相对简单。DFlash 的复杂度来自：

- block 是一段连续序列
- 需要维护下一批 tokens 的上下文
- 要保证 accept/reject 的 slice 语义正确
- rollback 不能破坏已接受的 prefix
- cache state 必须同步动起来

一旦 block 级 draft 出错，后面的生成状态会被污染，后续 token 会直接偏离正确轨迹。

---

## 十、DFlash 与 MTP / PLD / Drafter 的关系

这一层有一组非常重要的“同类机制”：

- DFlash
- MTP
- PLD
- Drafter

它们都属于 spec / draft 路径，但设计不同：

- PLD 更偏 history match / repetition 级别的 draft
- Drafter 更偏模板式或 assistant-level sidecar
- MTP 更偏模型自身头部 spec
- DFlash 更偏 block 级别的 assistant draft + verify

### 10.1 它们的共同点

都是在不让主模型做所有 token 计算时，提前预测和验证一段输出。

### 10.2 它们的不同点

- 机制来源不同
- 状态对象不同
- cache 处理方式不同
- acceptance 规则不同

所以它们不是替代关系，而是不同加速通道。

---

## 十一、DFlash 的关键实现特点总结

可以把 DFlash 的实现总结成几个关键点：

1. 它是 block-based speculative draft
2. 它使用 config contract 来决定是否启用
3. 它依赖 assistant forward / capture layers
4. 它维护 block K/V 和 draft cache
5. 它以 verify / partial-accept 为关键状态机
6. 它必须配合 scheduler 做 slot 与 cache 的正确控制

---

## 十二、最值得继续阅读的源码入口

如果你还要继续读 DFlash 的实现，建议顺序如下：

1. [src/dflash.zig](../src/dflash.zig)
2. [src/drafter.zig](../src/drafter.zig)
3. [src/generate.zig](../src/generate.zig)
4. [src/scheduler.zig](../src/scheduler.zig)
5. [src/transformer.zig](../src/transformer.zig)

重点看：

- `ForwardCtx.capture_layers`
- `DFlash2`
- `selectPath`
- `block_size`
- `accept` / `rollback` / `verify`
- `draft` / `cache.step` / `anchor row`

---

## 十三、结论

DFlash 是 mlx-serve 中非常典型的一条 spec draft 路径：

- 它不是普通 decode
- 它是 block-wise speculative generation
- 它依赖配置 contract 和 runtime 状态
- 它必须和 scheduler、cache、verify 流程协同
- 它是这套系统里“加速器”真正落地的一类机制

如果你已经理解了 DFlash，那么之后再看 MTP、PLD、Drafter 会发现它们都在同一条思路上，只是不同地方使用了不同的 speculation 设计。
