# mlx-serve transformer 与 generate 主链路详解（中文）

这篇文档继续沿着主链路往下走，解释真正的“算力层”是怎么工作的。

在前面的文档里，我们已经看到了：

- discovery 决定“模型是什么”
- model / loader 决定“模型怎么被装好”
- scheduler 决定“请求怎么被安排执行”

现在要看的是：

> 真正的 token 是如何从模型里生成出来的。

对应的关键文件主要是：

- [src/transformer.zig](../src/transformer.zig)
- [src/generate.zig](../src/generate.zig)
- [src/chat.zig](../src/chat.zig)
- [src/tokenizer.zig](../src/tokenizer.zig)

---

## 一、从“模型对象”到“前向计算”

在 mlx-serve 中，模型对象并不会直接输出文本。它首先要进入一个架构驱动的 forward 流程。

大致过程：

```text
prompt/tokenizer
   ↓
embedding / position / cache
   ↓
transformer.forward
   ↓
logits
   ↓
generate sampling
   ↓
next token
   ↓
循环直到 stop
```

所以真正的生成过程是：

- token 进入模型
- 在 transformer 内部执行层计算
- 得出 logits
- 由 generate 进行采样
- 产生下一 token
- 循环继续

---

## 二、transformer：模型架构的真正执行入口

[src/transformer.zig](../src/transformer.zig) 是最关键的执行层文件之一。

### 2.1 它不是“一个单纯的数据结构”

它既负责：

- 存储模型权重结构
- 维护 KV cache / 各类状态
- 根据模型架构分发到不同 forward 路径

也负责：

- 选择不同 attention / MoE / hybrid / dense 模式
- 处理量化、fused kernels、NAX 相关路径
- 处理 decode 与 prefill 的不同分支

### 2.2 它是“架构调度中心”

这也是为什么它很大。因为它同时要兼顾：

- GEMMA
- Qwen
- LFM2
- Nemotron
- DeepSeek V4
- H3 / mixed / vision / hybrid
- GGUF bridge

这些不同架构都要在这一层汇聚。

---

## 三、forward 的核心分工：prefill 与 decode

最重要的理解：

- prefill 和 decode 并不是同一个函数
- 它们的计算形态不同
- 它们使用的 cache 和 attention 机制也不同

### 3.1 prefill：构建上下文

prefill 主要用来：

- 将 prompt 处理成序列表示
- 形成 attention 需要的 key/value
- 预先填充上下文缓存

它重点是“立刻完成输入序列的全部上下文建立”。

### 3.2 decode：逐 token 生成

decode 只处理当前 token 或少量新 token，重点是：

- 更新 KV cache
- 计算下一 token logits
- 根据采样策略选下一个 token

这一阶段是实际生成过程的核心。

---

## 四、attention 与 KV cache：生成的最关键状态

在 LLM 里，真正让模型“记住前文”的核心，是 KV cache。

在 mlx-serve 中，KV cache 不是一个边角功能，而是生成系统的核心状态。

### 4.1 为什么 KV cache 重要

在生成时，模型不能从头重算所有历史，因为这样会非常慢。

所以系统维护一个缓存：

- key 记住历史上下文
- value 记住历史上下文
- 新 token 只需要增量更新

从逻辑上，这让模型具备“增量生成”的能力。

### 4.2 不是所有模型的 KV cache 都一样

不同架构可能有：

- 标准 attention
- sliding-window attention
- hybrid attention + SSM
- MoE + sparse attention
- MLA / GDN / linear attention

所以这里要持续处理架构差异，而不是统一成一个简单 cache 结构。

---

## 五、generate.zig：采样与生成循环的核心

真正生成 token 的控制器主要在：

- [src/generate.zig](../src/generate.zig)

它负责：

- 调用 transformer 的 forward
- 把 logits 变成候选 token
- 执行 top-k / top-p / temperature / repetition control
- 维护生成循环
- 检查是否结束

### 5.1 它是“生成状态机”

generate 不只是“做一次 forward”. 它更像一个状态机：

- 当前有多少 token 了
- 当前是否处于 prefill/decode
- 是否命中 repeating loop
- 是否需要 spec-decode
- 是否达到 max tokens / stop conditions

### 5.2 它在本项目里非常关键

因为这一步决定了：

- 模型是什么样的输出
- 什么时候结束
- 是否触发反复循环保护
- 是否启用 speculative draft 验证

---

## 六、采样：logits 转 token

采样逻辑不是简单的“argmax”。

项目支持的策略通常包括：

- temperature
- top-k
- top-p
- penalty / repetition guard
- stop tokens / grammar constraints

### 6.1 采样的关键不是“拿最大值”

而是：

- 生成自然文本
- 控制随机性
- 避免发散或重复输出
- 兼容 tool calling / structured output

特别是在 tool calling 场景中，采样的输出必须保证：

- 生成的 JSON/参数有效
- 语义能被下游 parser 正确识别
- 不因为分隔符或特殊白字符破坏结构

这也是为什么项目里 tool call 对格式要求很严。

---

## 七、speculative decoding 在 generate 中的作用

在 generate 层，speculative decoding 并不是一个“可选插件”，而是生成流程中的关键优化层。

### 7.1 draft + verify 的结构

典型流程：

1. small draft model / sidecar / deduced pattern 先给出若干候选 token
2. real model 进行校验
3. 只接受有效部分
4. 更新生成状态

### 7.2 它降低的是“单 token 的真实大模型调用次数”

这对长文本和长生成场景很重要，因为纯串行大模型 generate 会很慢。

---

## 八、loop-stop 与 repetition protection

在 generate 这层，除了采样之外，另一个非常核心的机制是：

- repetition detection
- loop-stop guard
- near-repeat filter

### 8.1 它不是“无关质量控制”

它直接决定输出是否被截断或者提前结束，避免：

- 一直重复同一段代码
- 反复生成同一个短句
- 增长上下文后陷入文本环

### 8.2 为什么它很重要

一个 LLM 如果质量和循环保护都没做好，最终表现会非常差。这个项目的实现明显把它当成生成逻辑的一部分，而不是事后修补。

---

## 九、chat 模板与 tokenizer：生成前的准备工作

真正的生成并不只靠 transformer，它还依赖：

- tokenizer
- prompt formatting
- chat template
- tool-call serialized prompt

这些在 [src/chat.zig](../src/chat.zig) 和 [src/tokenizer.zig](../src/tokenizer.zig) 里实现。

### 9.1 tokenizer 决定了 token stream 的真实形状

例如：

- 特殊 token 如何分割
- 数字分组规则
- 模型的 digit_group
- multi-turn 聊天如何编码

### 9.2 chat 模板决定了上下文语义

例如：

- ChatML
- Gemma template
- Llama-3 template
- Jinja2 template
- special reasoning tags

在模型的 prompt 组装时，template 的细节直接决定输出质量和 tool-call 是否能正确识别。

---

## 十、真正的“生成主循环”是什么样

如果用一个总流程概括，可以写成：

```text
build prompt tokens
   ↓
tokenizer.encode -> input_ids
   ↓
prefill build context + KV cache
   ↓
loop:
    model forward on current tokens
    logits -> sample next token
    append token to output
    update cache
    check stop / loop-stop / spec
    continue until complete
```

这就是整个生成主循环的本质。

---

## 十一、为什么这一层最容易踩坑

在调度/生成这层，最容易出 bug 的几个地方包括：

1. KV cache 失配
2. prefill/decode 状态不一致
3. loop-stop 检测过早或过晚
4. tool-call JSON 修正导致格式破坏
5. spec-decode 验证路径与普通 decode 不一致
6. prompt/template/tokenizer 细节导致输出改变

这些问题都不会出现在“接口层”，而是在真实生成流程中出现。

---

## 十二、从源码看最重要的两个入口

如果你真正要读源码，建议直接看这两个入口：

- [src/generate.zig](../src/generate.zig)
- [src/transformer.zig](../src/transformer.zig)

然后再配合：

- [src/chat.zig](../src/chat.zig)
- [src/tokenizer.zig](../src/tokenizer.zig)

这样你就能把：

- 模型输入如何变成 token
- token 如何进入 forward
- logits 如何采样
- 生成循环如何控制

这几件事串起来。

---

## 十三、结论

transformer 和 generate 是 mlx-serve 中真正意义上的“计算核心”。

它们把前面的模型发现、加载、调度全部落到：

- 具体层计算
- KV/state 管理
- 采样
- stop/loop guard
- speculative decode
- 最终生成输出

如果你理解了这两层，就基本能看懂这个项目的核心运行时了。

下一步最自然的延伸，是继续看：

- [src/chat.zig](../src/chat.zig) 的模板与 tool-call 处理
- [src/server.zig](../src/server.zig) 的协议桥接
- 以及特定架构实现，比如 qwen / lfm2 / dflash / mtp 等等。
