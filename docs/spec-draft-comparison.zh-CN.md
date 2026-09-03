# Drafter、PLD、MTP 的对比与协同（中文）

这一节把项目中最重要的三类 speculative generation 路径放到一起看：

- Drafter
- PLD
- MTP

它们各自解决的是生成加速中的不同问题，但它们共同属于同一条大线：

> 用“预测 + 验证”的方式减少大模型的实际计算量。

核心文件：

- [src/drafter.zig](../src/drafter.zig)
- [src/pld_index.zig](../src/pld_index.zig)
- [src/mtp.zig](../src/mtp.zig)
- [src/dflash.zig](../src/dflash.zig)
- [src/generate.zig](../src/generate.zig)
- [src/scheduler.zig](../src/scheduler.zig)

---

## 一、统一理解：它们都在做 speculative generation

先抽象一下：

- 大模型负责最终高质量验证
- 小模型 / sidecar / index / head 负责先估算未来 token
- 只接受被确认的 token
- 继续下一轮推理

这是 spec draft 的统一哲学。

不同实现只是：

- 预测来自哪里
- 验证怎么做
- cache 和 state 怎么维护
- 是否依赖外部表或模型自带头

---

## 二、Drafter：更偏“辅助模型/sidecar 路径”

Drafter 的定位偏“有专门的辅助生成路径”。

它通常意味着：

- 不是主 trunk 自带的普通 decode
- 而是某个旁路模型 / sidecar / 轻量模型负责先预测 token
- 然后由主模型确认

### 2.1 优点

- 逻辑上比较清晰
- 对主 trunk 的压力较轻
- 适合一些有专门辅助 head 的架构

### 2.2 局限

- 需要额外的 sidecar 配置
- 它的启用条件往往更依赖模型本身是否提供了 drafter
- 生成状态管理比普通解码多一层

### 2.3 语义上

Drafter 更接近：

> 一条“辅助生成器 + 验证器”的组合路径。

---

## 三、PLD：更偏“历史重复模式匹配”

PLD 是 Projected / Prefix Lookup / Pattern-Driven 的思路，或者更直白地说：

- 不是模型自己预测 token
- 而是从历史继续复用先前出现过的 n-gram / prefix 模式

它的核心是在：

- `pld_index.zig`
- `findMatch`
- `ngramRepeatScore`

等逻辑里，利用历史片段进行 match / repeat 推测。

### 3.1 它和模型的关系更“外部历史驱动”

PLD 更像：

- 模型本身不必先在每个 token 上产出非常强的草稿
- 直接根据旧上下文与已生成内容，找到重复模式，然后据此做预填充/预测

### 3.2 优点

- 很适合代码 / 结构化文本 / 重复模板
- 一些长上下文场景非常有用
- 不一定依赖一个单独的辅助模型

### 3.3 局限

- 它依赖重复模式的存在
- 对无明显重复模式的自然文本不一定高效
- 通常更接近“模式匹配预测”，不是“语义预测”

---

## 四、MTP：更偏“模型自带的预测头”

MTP 是最“模型原生”的 spec 分支。

它的特征是：

- 不是外部历史匹配
- 也不是单独的 sidecar generator
- 而是模型内部自己携带一个更专门的多 token 预测头

### 4.1 这意味着什么

它更像：

- 模型本身就设计了一个结构来做预测头
- 这个头与主干状态直接协同
- 生成时以多 token 为单位推进

### 4.2 MTP 的价值

它通常有两个明显好处：

- 更深地嵌入到模型结构中
- 生成中能更接近模型自己学习到的 multi-token 意图

### 4.3 它的复杂度也最高

因为它不只是“预测一个草稿”，而是：

- 需要主干状态
- 需要隐藏流/状态通道
- 需要 accept / verify / rollback
- 需要 slot isolation 与 cache management

---

## 五、三者的核心差异

| 机制 | 预测来源 | 典型特点 | 更适合的场景 |
|---|---|---|---|
| Drafter | 辅助 sidecar / draft model | 结构清晰，依赖辅助配置 | 模型有专门 draft 路径 |
| PLD | 历史模式与重复查找 | 更偏 pattern match | 代码、重复文本、长 context |
| MTP | 模型自带头部 | 最深度嵌入，状态复杂 | 高性能主链路 / 原生 spec |

简单说：

- Drafter = 旁路预测器
- PLD = 历史模式预测器
- MTP = 模型自带预测头

---

## 六、为什么它们会被同时放进同一条大线

因为它们都不是纯算力优化，而是：

- 最终还是要走 verify
- 仍然要管理状态和缓存
- 仍然要和 scheduler 协同
- 最终都要决定 accept 多少 token

换句话说：

> 它们都属于“生成前瞻 + 验证确认”的统一范式。

---

## 七、它们在 scheduler 中是怎样协同的

项目里 `scheduler.zig` 很关键，因为它看重的不是某个单一机制，而是：

- slot state
- spec mode
- enable / disable decisions
- partial accept
- block / draft commit
- cache rollback

### 7.1 scheduler 是统一控制器

它并不是只运行一个 spec 引擎，而是：

- 看当前 slot 是否允许 draft
- 看当前是否处于 MTP / dflash / pld / drafter 模式
- 看是否存在工具调用、语法约束、logprobs 等限制
- 看是否需要回滚一小段已写入的输出

### 7.2 也就是说

Drafter / PLD / MTP 不是互相竞争的“独立功能”，而是：

> 不同 spec 路径在统一 scheduler control 下运行。

---

## 八、它们如何共同影响生成吞吐

从高层解释：

- Drafter：提高候选生成效率
- PLD：提高历史复用和长序列重复生成效率
- MTP：把 speculation 集成到模型结构中

三者共同目标：

- 大模型不必对每个 token 都做全部 heavy forward
- 先用轻量或历史模式去猜测多个 token
- 验证器再决定接受多少

因此，工程上它们是一组互补能力，而不是彼此替代。

---

## 九、复杂度的上升顺序

如果按工程复杂度看，大致排序是：

1. PLD：偏历史模式匹配，结构较清晰
2. Drafter：需要 assistant / sidecar 路径和 config 接口
3. DFlash：更偏 block-draft + verify + cache control
4. MTP：最深度嵌入到模型结构和状态管理中

这也是为什么项目里会对它们分别放在不同文件和不同 runtime 阶段处理。

---

## 十、最值得继续读的源码入口

如果你继续深入，建议按这个顺序：

1. [src/pld_index.zig](../src/pld_index.zig)
2. [src/drafter.zig](../src/drafter.zig)
3. [src/dflash.zig](../src/dflash.zig)
4. [src/mtp.zig](../src/mtp.zig)
5. [src/scheduler.zig](../src/scheduler.zig)
6. [src/generate.zig](../src/generate.zig)

重点看：

- `findMatch`
- `ngramRepeatScore`
- `draft` / `verify` 流程
- `accept` / `rollback`
- `slotExclusiveDecode`
- spec mode in scheduler

---

## 十一、最终结论

Drafter、PLD 和 MTP 是 mlx-serve 里三条最重要的 speculative generation 分支：

- Drafter：辅助 draft / sidecar 路径
- PLD：历史重复模式预测
- MTP：模型本身自带的 spec 头

它们共同构成了项目中最关键的“生成加速层”，并都必须和 scheduler / cache / verify 机制协同工作。

如果你已经理解了这三者，那么你就基本把项目里“speculative decode”的骨架看明白了。
