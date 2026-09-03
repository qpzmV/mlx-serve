# mlx-serve 主流程源码导读（中文）

这篇文档是对项目主链路的进一步梳理，目的是让你能更快地从“文档认知”过渡到“源码阅读”。

如果把整个项目比作一条流水线，那么它的主链路可以概括成：

- 启动入口：解析参数，决定运行方式
- 模型发现：识别模型类型和加载方式
- 调度器：管理请求、队列、批处理和状态
- 模型前向：真正执行 token 生成
- HTTP/API：把请求协议转换成内部生成请求
- 返回客户端：流式输出或一次性输出

下面按照这个顺序讲。

---

## 一、主入口：main.zig + cli.zig

最上层入口是：

- [src/main.zig](../src/main.zig)
- [src/cli.zig](../src/cli.zig)

这两处文件的职责可以理解为：

1. 解析命令行参数
2. 确定当前是 CLI 模式、serve 模式、headless 模式，还是 app 触发模式
3. 决定是否加载模型、启动 HTTP 服务器、打开模型目录扫描
4. 处理 run / pull / list / serve / launch 等子命令

从设计上看，这一层不是“模型推理核心”，而是“总控和入口装配层”。

关键问题：

- `--model` 是单模型路径，还是目录扫描路径？
- `--serve` 是否立即启动 HTTP 服务？
- `--model-dir` 是否扫描一个模型目录并按需加载？
- 是否是 GGUF 路径、media 模型路径，还是标准 safetensors 模型？

这层决定了后续路径是如何走的。

---

## 二、模型发现：model_discovery.zig + model.zig

接下来核心的是：

- [src/model_discovery.zig](../src/model_discovery.zig)
- [src/model.zig](../src/model.zig)

这两部分负责把“磁盘上的模型”变成“可用的 LoadedModel / Runtime object”。

它们做的事情包括：

- 扫描目录中的模型
- 判断它是 `gemma`、`qwen`、`llama`、`gguf`，还是图片/音频/视频/3D media 模型
- 读取 `config.json` / safetensors / gguf metadata
- 生成权重/参数信息
- 发现模型的能力（text / vision / embedding / media / tool calling）

这里有两个非常关键的点：

### 1. 这是“发现”和“分类”的层

不是连接 GPU 计算的层，而是负责“知道这个模型是什么”的层。

### 2. 它决定后续走哪条 forward 路径

例如：

- 普通文本模型走 transformer 逻辑
- GGUF 走 embedded engine 逻辑
- media model 走 gen.zig 的统一生成层

所以它相当于项目的一个“分叉开关”。

---

## 三、架构分发：transformer.zig + chat.zig + gen.zig

真正的模型计算和协议行为的桥接在：

- [src/transformer.zig](../src/transformer.zig)
- [src/chat.zig](../src/chat.zig)
- [src/gen.zig](../src/gen.zig)

这几部分负责把“发现出来的模型”真正变成“可以生成内容”的对象。

### 1. transformer.zig：模型前向的核心

这是最核心的“算子/架构层”。

它负责：

- 路由到不同模型架构
- 处理 attention / MoE / hybrid / GatedDeltaNet 等结构
- 处理 KV cache、prefill 和 decode
- 处理各类量化、attention 方案和状态管理

它不是一个单一函数，而是一大套架构 dispatch 的集合。

### 2. chat.zig：聊天模板与工具调用

这是聊天协议层和模型输出层的关键文件。

它负责：

- 处理 chat template
- 处理 system / user / assistant / tool message
- 处理 reasoning / thinking
- 解析和修复模型生成的 tool calls
- 处理多种模板风格（ChatML、Gemma、Llama-3、Jinja 等）

所以这个文件是“模型输出格式与协议格式之间的桥梁”。

### 3. gen.zig：统一生成层

这是多模态和媒体生成的枢纽。

它管理：

- image generation
- audio generation
- video generation
- 3D generation
- LoRA 处理
- 模态检测与路由

也就是说，文本模型和 media 模型在这里被统一到一套生成框架下。

---

## 四、调度器：scheduler.zig

真正的“大脑”是：

- [src/scheduler.zig](../src/scheduler.zig)

这个层负责：

- 请求进入后如何排队
- 哪个 slot 是活跃的
- 是否需要并发 batch decode
- 什么时候进入 prefill，什么时候进入 decode
- 如何处理多个请求的 prefix cache
- 如何处理 speculative decoding 相关状态
- 是否在循环中触发 repetition loop stop

这个文件最重要的设计原则就在文档里反复强调：

> inference thread 是唯一的 MLX 调用者

也就是说，推理和 GPU 资源并不散落到每个连接线程，而是被集中到 scheduler/inference thread 这一条链路里。

这让它更容易：

- 控制并发
- 控制状态分配
- 避免 MLX 线程竞争
- 统一处理内存与批处理状态

所以，scheduler 是整个项目中“最像总控中心”的模块。

---

## 五、生成核心：generate.zig

生成逻辑的重点在：

- [src/generate.zig](../src/generate.zig)

这个文件负责：

- token 生成
- top-k/top-p/sampling
- speculative decoding（PLD / drafter / MTP）
- stop condition 处理
- KV cache 的推进与复用
- 回复结束的判定

可以把它看成“token 流生成发动机”。

这里有几个很关键的设计：

### 1. 生成不是简单的单步循环

它会根据请求状态判断：

- 是 prefill 阶段还是 decode 阶段
- 是正常生成还是 spec-decode 生成
- 需要验证还是直接提交

### 2. 生成会受缓存和批量状态影响

例如：

- prefix cache 命中
- 共享缓存重用
- 生成过程中的 loop-stop guard
- 采样策略和重复率感知

### 3. 它是运行时最复杂的一层

很多性能、稳定性和“坑”都集中在这一层。

---

## 六、HTTP 层：server.zig

接下来最重要的接口入口是：

- [src/server.zig](../src/server.zig)

它负责所有外部接入：

- `/v1/chat/completions`
- `/v1/messages`
- `/v1/responses`
- `/api/*`
- `/v1/models`
- media generation endpoint
- `/metrics`
- lan proxy

对使用者来说，这一层是“看得见的服务”；对开发者来说，它是“把协议翻译成内部请求”的层。

它的典型职责：

- 接收 JSON 或 multipart
- 解析字段
- 组织聊天消息结构
- 调起 scheduler 生成
- 处理 SSE / streaming 输出
- 把模型输出转换回标准 API 格式

也就是说：

- server.zig 是协议层
- scheduler.zig 是调度层
- transformer.zig 是模型层

它们三者是整个项目最重要的三层结构。

---

## 七、Ollama / Responses / Anthropic 层

除此之外，还有三层协议桥接：

- [src/ollama.zig](../src/ollama.zig)
- [src/responses.zig](../src/responses.zig)
- [src/chat.zig](../src/chat.zig)

它们的职责分别是：

### 1. ollama.zig

把 Ollama 请求转换成内部 OpenAI 风格结构，再把回复转换回 Ollama 格式。

注意：这不是“另一套推理代码”，而是“协议翻译层”。

### 2. responses.zig

处理 OpenAI Responses API 的状态管理、响应结构、sequence number 和 stateful chain。

它重点关注：

- `previous_response_id`
- SSE event seq
- stateful response store
- WebSocket transport

### 3. chat.zig

既是聊天模板解析，也负责 tool call 和 reason/think 的结构化处理。

这非常关键，因为 tool calling 相关 bug 往往不是推理模型本身的问题，而是“输出格式被错误解析或截断”。

---

## 八、从请求到输出的主流程示意

可以把整个主链路压缩成下面这个流程：

```text
用户请求
   ↓
server.zig 解析协议请求
   ↓
转换成内部 Message / Request / Slot 结构
   ↓
scheduler.zig 分配 slot + queue + batching
   ↓
transformer.zig 走对应模型架构 forward
   ↓
generate.zig 做 sampling / stop / spec-decode
   ↓
输出 SSE / WS / JSON / tool call / usage
   ↓
返回客户端
```

这条线几乎覆盖了几乎所有核心逻辑。

---

## 九、最容易卡住的“理解切入点”

如果你刚接触这个项目，最容易迷路的地方通常不是模型算子，而是下面几个层面：

### 1. 协议层和推理层混在一起看

很多代码都在 server.zig 里混合处理协议和生成状态，容易让人误以为“所有东西都在这一个文件里”。

正确理解应该是：

- 协议层：server / ollama / responses
- 调度层：scheduler
- 模型层：transformer / generate

### 2. 模型类型识别非常关键

在 model_discovery.zig 里，模型的类型决定后续走哪个分支。很多问题在这里就被切掉了。

### 3. 生成并不单纯是一个 for loop

它受：

- batching
- prefix cache
- speculative decoding
- memory preflight
- tool calling
- streaming buffer

等状态共同影响。

### 4. gotchas 是源码理解的关键补充

如果你直接看源码，很多地方看起来像“很奇怪的条件判断”，但在 gotchas 里会解释：

- 为什么之前是这样写的
- 为什么一定要这样约束
- 这条路径为什么会在真实环境里出问题

---

## 十、最推荐的阅读顺序（源码视角）

如果你按源码阅读，我建议这样读：

1. [src/main.zig](../src/main.zig)
2. [src/cli.zig](../src/cli.zig)
3. [src/model_discovery.zig](../src/model_discovery.zig)
4. [src/model.zig](../src/model.zig)
5. [src/scheduler.zig](../src/scheduler.zig)
6. [src/server.zig](../src/server.zig)
7. [src/transformer.zig](../src/transformer.zig)
8. [src/generate.zig](../src/generate.zig)
9. [src/chat.zig](../src/chat.zig)
10. [src/ollama.zig](../src/ollama.zig)
11. [src/responses.zig](../src/responses.zig)

如果只看一个功能，例如 tool calling，那么应该补看：

- [src/chat.zig](../src/chat.zig)
- [docs/gotchas/tool-calling.md](gotchas/tool-calling.md)

如果想看性能与 kernel：

- [src/transformer.zig](../src/transformer.zig)
- [docs/gotchas/engine-mlx.md](gotchas/engine-mlx.md)

如果想看 HTTP 和请求处理：

- [src/server.zig](../src/server.zig)
- [docs/gotchas/server-http.md](gotchas/server-http.md)

---

## 十一、结论

从源码结构看，mlx-serve 不是一个“单一功能的小服务器”，而是一个分层清晰的本地 AI 推理平台。它的核心层次是：

- 入口层：main / cli
- 发现层：model_discovery / model
- 调度层：scheduler
- 生成层：generate
- 模型层：transformer
- 协议层：server / ollama / responses
- 格式层：chat / tool calling

理解这条主链路之后，再去看细节文件会非常容易，因为你已经知道：

- 这个文件属于哪一层
- 它解决的是什么问题
- 它和上游/下游文件是什么关系

这也是阅读这个项目最重要的能力。

如果你真正要继续往下读代码，我建议下一步直接从：

- [src/main.zig](../src/main.zig)
- [src/model_discovery.zig](../src/model_discovery.zig)
- [src/scheduler.zig](../src/scheduler.zig)

开始逐段读，效果会比直接从某个函数入手好很多。
