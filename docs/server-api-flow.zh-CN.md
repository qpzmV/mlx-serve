# mlx-serve server 与协议层详解（中文）

这是整个主链路中最容易“看起来像接口层，但其实决定运行行为”的一层。

前面的文档已经讲了：

- discovery：模型是什么
- model：模型如何装好
- scheduler：请求如何执行
- generate/transformer：真正如何生成 token

现在要理解的是：

> HTTP / SSE / WebSocket / OpenAI / Anthropic / Ollama 这些协议入口，怎么把外部请求转换成内部生成任务。

关键文件：

- [src/server.zig](../src/server.zig)
- [src/ollama.zig](../src/ollama.zig)
- [src/lan.zig](../src/lan.zig)
- [src/responses.zig](../src/responses.zig)
- [src/ws.zig](../src/ws.zig)
- [src/gen_sse.zig](../src/gen_sse.zig)

---

## 一、server.zig 是“协议入口 + 控制中心”

[src/server.zig](../src/server.zig) 是整个系统中最重要的协议入口之一。

它并不只负责“接受请求”，而是控制：

- 路由解析
- 请求 body 解析
- 认证和鉴权
- 模型识别和挂载
- 多协议适配
- 生成结果如何流式返回
- 并发、超时、错误处理

可以把它理解成：

> 连接外部世界和内部运行时的总接口层。

---

## 二、协议层最重要的职责：把外部请求转成内部任务

外部请求通常会带来很多不同的格式：

- OpenAI chat/completions
- OpenAI responses
- Anthropic messages
- Ollama API
- LAN share / peer proxy
- SSE / WebSocket 流式输出

这些格式差异很大，但内部 system 需要统一成一个生成任务。

所以 server 的工作核心是：

```text
外部协议请求
   ↓
统一请求对象
   ↓
绑定 model + generation config
   ↓
交给 scheduler
   ↓
返回协议格式输出
```

这说明 server 其实是“协议转换器 + 调度入口”。

---

## 三、OpenAI 兼容层：核心的兼容接口

这个项目的文档里极其明确：它支持 OpenAI 兼容的 API。

包括：

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/responses`
- `/v1/models`
- `/v1/embeddings`
- `/v1/images/*`

### 3.1 它的要求不是“只像”，而是“兼容语义”

这意味着：

- field names
- usage format
- `reasoning_content`
- tool call structure
- stop reason
- stream chunk semantics

它们都必须遵循 OpenAI 的基本约定，或者在 project 自己的扩展上保持兼容。

### 3.2 也是很多 bug 的集中区域

因为协议层最容易出现：

- 请求字段解码错
- tool call 参数修正错
- stream chunk 类型不一致
- usage 字段丢失
- 生成结束状态不一致

所以它也是项目最容易出“看起来能跑，但接口语义不对”的地方。

---

## 四、Anthropic 与 Ollama：协议适配不是简单转发

#### 4.1 Anthropic `/v1/messages`

Anthropic 的 messages API 结构和 OpenAI 很不一样：

- typed blocks
- content array
- input schema / parameters
- stop reason mapping
- stream lifecycle

这个项目不只是简单做 JSON 转发，而是需要把 Anthropic 的语义真实适配到内部生成任务格式。

#### 4.2 Ollama `/api/*`

Ollama 也是一个很大的兼容层。其特点是：

- 还是 HTTP 接口
- 但有自己一套格式
- 很多 endpoint 是 translation，不是直接走原生后端

它通常在 [src/ollama.zig](../src/ollama.zig) 里处理，重点是：

- model name resolution
- tag 处理
- SSE 到 NDJSON 转换
- `/api/generate`、`/api/chat` 等入口适配

这体现了一个设计原则：

> 兼容层是“协议翻译”，不是“接口封装”。

---

## 五、Responses API 与 structured output

[src/responses.zig](../src/responses.zig) 是一个比较重要的模块。

它的作用不是简单的“另一个接口”，而是：

- 处理 stateful response
- 维护 ResponseStore
- 处理 event sequence
- 处理 streaming event envelope
- 根据 request 组装并返回统一的 response 格式

### 5.1 它强调“状态管理”

Responses API 更容易表现出 stateful 行为，而且对 event 的 ordering 和 sequence_number 要求严。

这意味着：

- 每个响应不是临时对象
- 要保留状态
- 事件流要有顺序
- background task / continuation 并不一样

### 5.2 它和 chat/completions 之间的区别很明显

OpenAI chat/completions 更偏“短流程”，而 responses 更偏“状态感强的 structured 生成任务”。

---

## 六、SSE / WebSocket：流式输出的真正核心

项目不只有 HTTP JSON 返回，也支持流式传输：

- SSE
- WebSocket

关键文件：

- [src/gen_sse.zig](../src/gen_sse.zig)
- [src/ws.zig](../src/ws.zig)

### 6.1 流式输出不是“调用一次就直接返回”

而是：

- 生成 token
- 进入 chunk encoder
- 分发到 SSE / WS
- 维持 keepalive
- 在需要时附加 usage / finish 信息

### 6.2 这层的重点是“事件语义”

尤其是：

- 开始事件
- delta 事件
- reasoning 事件
- tool call 事件
- usage 事件
- [DONE] / finish 标识

这些都必须严格保持协议顺序。

如果顺序错了，客户端会直接把生成内容解释错。

---

## 七、LAN 共享：不是普通客户端接口，而是一个特殊 transport

[src/lan.zig](../src/lan.zig) 处理的是局域网共享网络层。

它的定位非常明确：

- 不是业务模型逻辑
- 不是聊天功能
- 是“网络层代理 / 共享 / 转发”机制

### 7.1 它做的是“transport”而非“protocol semantics”

也就是说，它主要负责：

- discovery / peer route
- tunnel / proxy
- shared set / allowlist
- peer auth / routing class
- loop prevention

### 7.2 这类逻辑很容易被忽略

但它的意义非常大，因为它决定了：

- 能否在 LAN 内共享模型
- 能否跨机器调用
- 是否存在 loop / self-fetch / invalid route

---

## 八、server 中的“统一生成任务”是怎么形成的

从协议层看，server 最终要做的事情可以概括为：

```text
协议请求 body
   ↓
decode into internal request struct
   ↓
resolve model + generation params
   ↓
tokenize / inspect tool schema / images / media
   ↓
call scheduler.submit
   ↓
return stream or final result
```

这说明：

- server 不只做请求过滤
- 它其实是“统一请求形态”的生产者
- scheduler 只看到标准化后的内部请求

这样才能让不同协议接入到同一套生成逻辑中。

---

## 九、错误处理与资源约束也在这里做

协议层需要处理很多“运行前检查”与“运行中检查”问题，例如：

- 模型不存在
- context 超长
- media 类型不支持
- image/audio/video 维度错误
- tool schema 无效
- memory insufficient
- load error / 503 / 400 / 404

如果这些错误不在 server 层做好，后面的 scheduler / generate 会出现逻辑错误而不是协议层错误。

---

## 十、最容易踩坑的几个点

### 1. 协议格式和内部格式不对齐

最典型的 bug 是：

- OpenAI 兼容字段被解析错
- 封装层和内部层语义不一致
- stream 与 non-stream 输出内容不一致

### 2. tool call 解析和协议层交错

tool call 需要在 protocol 层解析，再交给 scheduler / transform 执行。

### 3. stream lifecycle 处理

SSE/WS 的 chunk 顺序、finish 语义和 usage 情况特别容易出错。

### 4. media request 和文本 request 混用

例如 image/audio/video 请求必须被识别成 media route，而不是文本生成 route。

---

## 十一、建议的源码阅读顺序

如果你要继续对照源码，最好的路径是：

1. [src/server.zig](../src/server.zig)
2. [src/ollama.zig](../src/ollama.zig)
3. [src/responses.zig](../src/responses.zig)
4. [src/gen_sse.zig](../src/gen_sse.zig)
5. [src/ws.zig](../src/ws.zig)
6. [src/lan.zig](../src/lan.zig)

这样你就能顺着：

- 请求进入
- 协议解析
- 内部统一请求
- scheduler 提交
- 生成输出
- 协议返回

这条主链全部串起来。

---

## 十二、结论

server 层是 mlx-serve 与外部世界之间的核心桥梁。

它承担的角色不是“单纯给接口”，而是：

- 规范协议输入
- 转成内部运行时任务
- 处理认证、模型解析、媒体支持、streaming
- 在适配不同 API 风格时保持统一语义

它和 scheduler、generate、transformer 一起，构成整个系统的三层：

- 外部协议层：server
- 执行调度层：scheduler
- 算力生成层：transformer / generate

这三层一起决定了整个系统是如何完成一次真正的推理。 
