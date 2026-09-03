# mlx-serve 调度与请求执行链路详解（中文）

这篇文档继续沿着项目主链路往下走：

- 模型被发现并加载
- 请求进入 server
- scheduler 接收请求
- 进入 batch / prefill / decode / spec-decode / loop-stop
- 最终输出结果给客户端

如果说 discovery 和 model 是“准备模型”，那么 scheduler 就是整个系统真正的“运行中枢”。

---

## 一、scheduler 在项目中的位置

最关键的文件包括：

- [src/scheduler.zig](../src/scheduler.zig)
- [src/server.zig](../src/server.zig)
- [src/generate.zig](../src/generate.zig)
- [src/transformer.zig](../src/transformer.zig)

它们之间的关系可以概括为：

```text
server 接收 HTTP 请求
   ↓
解析 request / model / prompt / tools / media
   ↓
scheduler 维护请求队列与 slot
   ↓
transformer / generate 执行真实推理
   ↓
返回流式或非流式输出
```

也就是说：

- server 是“协议层”
- scheduler 是“执行层中枢”
- generate / transformer 是“算力层”

---

## 二、为什么 scheduler 很重要

在本项目中，scheduler 不是一个简单的“按顺序执行任务”的对象，而是一个复杂的运行时调度器。

它需要处理：

- 多个请求同时到达
- 不同请求的模型和 context 长度不同
- 是否允许 batch decode
- 是否需要预填充（prefill）
- 是否需要 spec-decode / MTP / DFlash / drafter
- 是否要做 loop-stop / repetition detection
- 是否要给出资源约束和 memory guard

所以 scheduler 是整个 runtime 的“总控结构”。

---

## 三、请求进入后，先发生什么

在 server 层收到请求后，最重要的动作不是立刻推理，而是：

1. 解析请求协议
2. 绑定 model
3. 处理 prompt / tool call / image / audio / video 等输入
4. 确认该请求何时进入 scheduler
5. 生成一个 slot / task / generation state

### 3.1 这是“请求对象”阶段

请求一旦进入 runtime，通常会有：

- prompt / input tokens
- model id / model handle
- context length
- output budget
- generation config
- stream flag
- tool setting
- media fields

这些构成一个完整的“生成任务”。

### 3.2 不是直接把 token 送给模型

注意：

> scheduler 负责“把任务组织成可执行的生成单元”，但真正的数学计算仍然由 transformer / generate 实现。

所以这不是“一个函数直接生成 token”，而是“请求被拆成 slot + pipeline + state，然后再推进”。

---

## 四、slot：最核心的概念之一

`slot` 在 scheduler 中非常关键，因为这个系统是多请求并发/多 generation 并发的。

一个 slot 通常代表：

- 一个活跃请求
- 一个上下文状态
- 一个当前生成位置
- 一套缓存状态（KV / prefix / spec）
- 一个输出流

### 4.1 slot 不是“简单的线程对象”

它可能同时持有：

- cache state
- prefill 状态
- decode 状态
- loop-stop 观察状态
- dedup/evict/restore 信息

因此，slot 实际上是一份完整的生成任务上下文。

---

## 五、prefill 与 decode 是最关键的两步

项目中的运行方式通常分成：

- prefill：处理输入 prompt，建立上下文状态
- decode：根据当前状态逐 token 生成输出

### 5.1 prefill 是“建立上下文”

这一步通常处理：

- prompt embedding
- attention 前缀构建
- KV cache 形成
- 生成初始 hidden state

prefill 的性能敏感点包括：

- chunk size
- memory budget
- kv cache size
- hybrid / sliding-window / MoE 等特殊结构

### 5.2 decode 是“持续生成”

decode 是后续生成的核心：

- 从已有 KV cache 继续输出 token
- 逐 token 更新状态
- 根据 sampling / top-k / top-p 选择下一 token
- 可能调用 spec-decode 进行加速

这也是最能体现“模型 runtime”本质的阶段。

---

## 六、batch decode：并发生成时的关键机制

项目高度强调：

- concurrent requests are batched
- decoding uses shared attention / shared caches where possible
- `--max-concurrent` is not a decode gate, but queue sizing

这是很关键的一点：

### 6.1 scheduler 不只是串行处理请求

它允许多个请求同时进入 decode 阶段，但它仍然会维持 batched decode 的约束。

这意味着：

- 请求可以共享 decode 时间窗口
- 但不是所有请求都能无限并发
- 需要在 padding waste、context length、memory 等条件下做决策

### 6.2 这里的难点是“batch 成本控制”

项目有明确的规则：

- batched decode group 是根据 padding waste 约束控制，而不是简单按 slot 数
- 它只保留“较大前缀”并让长尾请求回退到 serial

这说明调度策略不是“越多越好”，而是“在内存 / 计算 / padding waste 三者之间做平衡”。

---

## 七、spec-decode：加速生成的重要机制

这是这个项目里特别明显的一层：

- PLD
- drafter
- DFlash
- MTP
- DSpark

这些东西都属于“speculative decoding / speculative generation”范围。

### 7.1 为什么要做 spec-decode

它的核心意思是：

- 不要让每一个 token 都等真正大模型独立计算
- 先用一个小模型或 draft 结构预测多个候选 token
- 再由目标模型确认这些 token 是否真的合法

从高层角度讲：

> 让“生成更快”，但不损失本质生成质量。

### 7.2 不是所有模型都启用相同的 spec 机制

不同架构可能使用：

- MTP
- DFlash
- PLD
- drafter
- hybrid 路径

而且还有优先级和条件控制：

- tools / grammar / logprobs 可能禁用某些 spec path
- 某些模型的 spec sidecar 只在对应架构上生效
- 有些模型的 draft mode 只在明确支持时才开启

### 7.3 scheduler 负责协调 spec 流程

scheduler 不是直接实现 spec-draft 的数学逻辑，而是负责：

- 哪个 slot 允许 spec
- 当前是否处于 accept / reject / verify 状态
- 什么时候切回正常 decode
- 什么时候结束 draft round

---

## 八、loop-stop：避免重复输出循环

这是另一个非常关键的执行控制机制。

项目中明确有：

- near-repeat loop detection
- repetition loop stop
- loop-stop reason
- finish_reason = "length"
- `finish_details: {"type":"repetition_loop"}`

### 8.1 意思是什么

模型在生成时如果出现：

- 内容重复很多
- 结构相似
- 4-gram 或 novelty 比例异常

那么 scheduler 会判定为一个“循环生成”问题，并提前终止当前生成。

### 8.2 这是一个“执行质量守卫”

它不是业务层逻辑，而是生成阶段的质量防护手段。

它确保：

- 不会不停重复生成同样的内容
- 不会在无穷循环中卡住
- 生成结果有明确的终止语义

---

## 九、工具调用在调度链路里的角色

工具调用并不是简单的“插件功能”，它必须在生成执行过程中参与决定：

- 解析工具调用
- 合法性修正
- 解析 JSON 参数
- 结构化输出
- 触发后续 tool execution

### 9.1 它会影响生成控制

工具调用经常会：

- 暂停普通文本生成
- 触发 function call 的解析
- 让它看起来像另外一个中间状态
- 触发 prefix cache invalidation

也就是说：

> tool call 不是 “额外功能”，而是生成状态转换的一种重要事件。

### 9.2 调度层必须知道工具事件

因为 scheduler 负责 slot / cache / generation status 的连续性，工具调用会直接影响：

- 当前上下文是否仍然有效
- 缓存是否需要放弃
- 是否应进入特殊的工具 call 生成分支

---

## 十、流式输出与非流式输出如何走

server 最终要对用户返回结果，但 scheduler 需要区分：

- stream = true
- stream = false

### 10.1 stream 模式

在 stream 模式中，输出不是等整段生成完再返回，而是：

- token-by-token 发送
- 需要处理 SSE / WS / chunk framing
- 需要维持 keepalive / heartbeat
- 需要对 tool call / reasoning / content 分流

### 10.2 non-stream 模式

non-stream 就是等整个回复生成结束，然后一次性发回。

不过这里有一个非常重要的设计原则：

> stream 和 non-stream 输出必须保持一致的内容语义，只有传输方式不同。

这也是项目里反复强调的“字节级一致性原则”。

---

## 十一、cache 与 memory 管理在 scheduler 中的意义

真正的推理线程是单一的，且是线程内核中调用 MLX 的唯一地方。

所以 scheduler 需要管理：

- prefix cache
- KV cache
- disk cache
- memory pressure
- slot eviction
- hot cache restore

### 11.1 这是保证系统稳定性的关键

如果没有正确的 cache / memory 管理：

- 长请求会挤占短请求
- / prefix cache 会误用旧状态
- 请求越积越多，最终出现 OOM 或错误复用

### 11.2 scheduler 不是只调度 token

它还在调度“内存资源”。

也就是说：

> 它不仅负责“什么时候跑”，还负责“跑之前是否有资格跑”。

---

## 十二、常见主线思维：从请求到 token 的串联

把调度链路压缩成一句话：

```text
request -> slot -> prefill -> decode -> batch/spec/loop-stop -> output
```

如果要更细一点：

```text
server parse request
  -> bind model
  -> scheduler create slot
  -> prefill build KV/context
  -> decode generate tokens
  -> optional spec-decode / tool call / streaming
  -> loop-stop / completion check
  -> return response
```

---

## 十三、最适合的源码阅读顺序

如果你继续读源码，建议按这个顺序：

1. [src/server.zig](../src/server.zig)
2. [src/scheduler.zig](../src/scheduler.zig)
3. [src/generate.zig](../src/generate.zig)
4. [src/transformer.zig](../src/transformer.zig)
5. [src/chat.zig](../src/chat.zig)

这样你可以从“请求进入”一路追到“真正的 token 生成”。

---

## 十四、结论

scheduler 在 mlx-serve 中并不是一个“上层管理器”那么简单，它是整个 runtime 的中枢。

它同时处理：

- 请求优先级
- batch / serial split
- prefill / decode 转移
- spec-decode 协调
- memory / cache state
- loop-stop / quality guards
- tool-calling / streaming output

如果说 discovery 决定模型是什么，scheduler 决定“这个模型如何按请求被真正执行”，那么它就是整个项目运行流程里最关键的一层之一。

下一步最自然的延伸，就是继续看真正的生成实现：

- [src/generate.zig](../src/generate.zig)
- [src/transformer.zig](../src/transformer.zig)

因为那才是“token 真正如何被产生”的底层实现。 
