# mlx-serve 源码主流程深度导读（中文）

这篇文档是对整个项目的“主链路深度拆解”，重点不是讲某个函数，而是讲“代码是怎么串起来跑起来的”。

如果前面的文档是地图，那么这篇文档就是路线图。它会围绕下面这条主线展开：

- 启动入口
- 模型发现与加载
- 请求进来
- 调度与 batch
- 进入模型 forward
- 生成 token
- 输出 SSE / JSON / tools

---

## 一、先看全图：项目的主链路是什么

整个项目可以抽象成下面这条主线：

```text
main.zig / cli.zig
    ↓
模型发现（model_discovery.zig）
    ↓
模型加载（model.zig / transformer.zig）
    ↓
请求接入（server.zig）
    ↓
调度与并发（scheduler.zig）
    ↓
模型 forward（transformer.zig / generate.zig）
    ↓
流式输出 / tool calling / usage / finish_reason
    ↓
客户端收到结果
```

这个主链路的关键特征是：

- 入口层负责启动和参数解析
- 模型层负责发现、加载和分发架构
- 调度层负责并发和状态管理
- 推理层负责真正生成 token
- 协议层负责把结果发回客户端

---

## 二、启动入口：main.zig 与 cli.zig

最先触发的是：

- [src/main.zig](../src/main.zig)
- [src/cli.zig](../src/cli.zig)

这一层做的事情非常清晰：

1. 解析命令行参数
2. 判断运行模式
3. 构造全局配置
4. 决定是否启动 HTTP server
5. 决定是否立刻加载模型

### 典型运行模式

- `mlx-serve --model ... --serve`
- `mlx-serve --model-dir ... --serve`
- `mlx-serve run <model>`
- `mlx-serve pull <model>`
- `mlx-serve serve`
- `mlx-serve launch <agent>`

这些模式实际上都在同一套总流程下工作，只是入口和目的不同。

### 入口的核心意义

cli 层不是“算子层”，它是“总控层”。

它负责：

- 拿到模型路径
- 确定是本地 model 还是目录扫描
- 触发 discovery
- 决定 server 是否启动
- 决定是否进入 one-shot 交互模式

换句话说，main/cli 是整个项目的总开关，不是推理核心。

---

## 三、模型发现：model_discovery.zig

真正的“模型是什么”判断主要集中在：

- [src/model_discovery.zig](../src/model_discovery.zig)

这个文件非常重要，因为它决定：

- 是标准 MLX 模型
- 还是 GGUF
- 还是 media 模型
- 还是一个需要走统一生成层的模型

### 模型发现做了什么

它会检查：

- 目录里有没有 `config.json`
- 有没有 safetensors
- 有没有 GGUF
- 是不是 `model_type` 对应的支持架构
- 是不是一个图片、音频、视频、3D 模型

它的意义不是“跑模型”，而是“知道这个模型要走哪条路”。

### 这一步的难点

因为模型类型很多，且很容易出现“看起来像某个架构，实际上不是”，所以 discovery 的设计非常重：

- 不同模型类型需要不同 forward path
- GGUF 和 safetensors 不是同一个逻辑分支
- media model 也不是 chat model 的同一路径

如果 discovery 错了，后面的加载、查找、推理都会错，所以它是第一道关键门。

---

## 四、模型加载：model.zig + transformer.zig

模型加载并不只是“读权重”，而是把模型对象构造出来，并绑定好其对应架构。关键文件：

- [src/model.zig](../src/model.zig)
- [src/transformer.zig](../src/transformer.zig)

### 4.1 model.zig：构造模型对象

它负责：

- 解析参数
- 读取内置格式
- 构造模型配置
- 生成 `LoadedModel`
- 绑定它的 tokenizer、weights、engine 和 runtime state

### 4.2 transformer.zig：真正的架构分派

它是“数学计算和架构分发”的核心。

它会根据 `model_type` 来选择：

- 普通 Transformer
- 混合模型
- MoE 模型
- GatedDeltaNet / hybrid / Mamba 类架构
- media 模型
- Vision / M-RoPE / SigLIP / qwen vision 等特殊结构

这也是为什么它在整个项目里非常大，而且非常关键。

### 核心逻辑

加载好模型之后，后面真正的 forward 并不在 server 层，而是在 transformer / generate 这两层中间完成。

---

## 五、HTTP 协议入口：server.zig

用户真正打到服务端的是：

- [src/server.zig](../src/server.zig)

它负责：

- 接收 HTTP 请求
- 做协议解析
- 组装聊天消息结构
- 触发请求执行
- 返回流式 / 非流式结果

### 5.1 协议层的职责

这层不是“做推理”的层，而是“承接外部请求并把它翻译成内部请求”的层。

它处理以下模式：

- OpenAI chat/completions
- Anthropic /v1/messages
- OpenAI Responses
- Ollama /api/*
- media generation endpoints
- /v1/models
- /health
- /metrics

### 5.2 它如何和 scheduler 连起来

server 层不会直接跑大规模 forward，它会把请求包装进内部对象，并交给 scheduler 执行。

所以可以把 server 理解为：

- 一边连接外部协议
- 一边把请求发入推理调度

---

## 六、调度器：scheduler.zig

真正管理运行时状态的是：

- [src/scheduler.zig](../src/scheduler.zig)

这个层的职责非常重，它相当于整个项目的“runtime center”。

### 6.1 它管理什么

- 请求 slot
- 批处理
- 并发处理
- prefill / decode 分阶段
- KV cache
- prefix cache
- 生成状态和异常状态

### 6.2 为何它很关键

因为这项目非常强调：

> inference thread is the sole MLX caller

也就是说，真正的 GPU / MLX 调用必须集中在一个线程里，避免多个线程同时操作 MLX 对象导致乱序或崩溃。这是项目中一个非常关键的工程设计。

### 6.3 调度器的本质

它本质上是：

- 任务总控
- 请求生命周期管理
- 推理状态管理
- 资源调度管理

它不是“模型内部实现”，但它决定了“整个系统如何稳定地跑起来”。

---

## 七、生成核心：generate.zig

真正的 token 生成核心在：

- [src/generate.zig](../src/generate.zig)

这层做的事情包括：

- 采样（temperature / top-p / top-k）
- 生成 token
- stop condition
- loop-stop / repetition loop
- speculative decoding
- 处理 MTP / drafter / PLD 之类的加速方法

理解这一层最重要的是：

> 生成不是一个单纯的 while loop，它是“调度 + state + sampling + cache + validation”组合起来的运行时状态机。

### 7.1 prefill 与 decode

模型运行通常分两段：

- prefill：处理输入上下文
- decode：逐 token 生成

prefill 和 decode 会触发不同的状态和不同的缓存行为。

### 7.2 spec-decode 与 draft

这个项目很强调 spec-decode，例如：

- PLD
- drafter
- MTP

这些都在 generate.zig 这一层发挥作用。

也就是说，性能优化并不在 server 层，而是在生成引擎层。

---

## 八、聊天与工具调用：chat.zig

这个文件是很容易被忽视，但非常关键的：

- [src/chat.zig](../src/chat.zig)

它负责：

- 聊天模板
- tool call 识别
- reasoning / thinking block
- 输出格式整理
- 修正损坏的 JSON / XML tool call

### 它的工程含义

很多模型输出并不“天然就符合 OpenAI 工具调用格式”，所以这里需要做：

- 严格解析
- 宽松修复
- 结构修正
- schema coercion

这也是为什么文档里会特别强调“tool calling”这类 gotchas。

### 关键判断

这层是“数据输出被重构成客户端可消费数据”的地方。它不是模型计算层，但它决定最终用户是否能用工具调用正确工作。

---

## 九、Ollama / Responses / Anthropic：协议桥接层

额外的协议桥接在：

- [src/ollama.zig](../src/ollama.zig)
- [src/responses.zig](../src/responses.zig)

### 9.1 ollama.zig

它将 Ollama 的协议请求翻译成内部统一的 OpenAI-like request，反过来把内部结果也翻译回 Ollama 格式。

它非常重要，因为项目主打“兼容 Ollama ecosystem”。

### 9.2 responses.zig

它处理更现代的 OpenAI Responses API，同时支持：

- stateful response chain
- SSE
- sequence number
- WS transport
- previous_response_id

### 9.3 这两个层的定位

它们都不是推理引擎，都是“协议接品器”。

也就是说，真正的计算仍然走上面的 scheduler + generate + transformer，而这些协议桥接只是把它包装成兼容格式。

---

## 十、最重要的理解：这不是单一线程的传统服务

这个项目很明显设计为：

- 一个进程内承载多模型
- 服务端协议统一
- 推理线程集中
- 生成状态是一个运行时状态机

所以阅读它时，千万不要把它看成传统的“HTTP server + model library”这么简单。它实际上像：

- 一个统一的本地 AI runtime
- 一个协议适配层
- 一个多模型调度器
- 一个语义生成引擎

这也是它为什么文档会强调：

- scheduler
- cache
- prefix cache
- batch
- spec decode
- media generation
- tools parsing
- LAN proxy

这些地方都不是边角，而是主线。

---

## 十一、从源码阅读的最佳入口

如果你现在要看源码，最推荐的顺序是：

1. [src/main.zig](../src/main.zig)
2. [src/cli.zig](../src/cli.zig)
3. [src/model_discovery.zig](../src/model_discovery.zig)
4. [src/model.zig](../src/model.zig)
5. [src/server.zig](../src/server.zig)
6. [src/scheduler.zig](../src/scheduler.zig)
7. [src/transformer.zig](../src/transformer.zig)
8. [src/generate.zig](../src/generate.zig)
9. [src/chat.zig](../src/chat.zig)
10. [src/ollama.zig](../src/ollama.zig)
11. [src/responses.zig](../src/responses.zig)

这样顺着读，你不会一上来就卡在某个深层 kernel，同时又能抓住主线。

---

## 十二、结论

这套项目的真正主线可以概括成一句话：

> 入口解析模型与协议，调度器管理请求状态，transformer + generate 负责真正推理，server 层负责协议输出，而 chat / tool-calling / responses / ollama 则负责把运行时结果包装成不同客户端能理解的格式。

如果你能把这条主线想清楚，再去看具体代码，就不会再被“一个函数突然跳出来、另一个文件又在讲协议、还有一个文件在讲 cache”这种信息流打断。

最重要的不是记住每个文件细节，而是知道：

- 这个文件属于哪一层
- 它负责什么
- 它和前后文件是什么关系

这才是正确的源码阅读方式。
