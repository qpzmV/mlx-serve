# mlx-serve 模型发现与加载流程详解（中文）

这是项目中非常关键的一层。很多人一上来就看 server 或 generate，但真正决定“这个模型能不能被正确跑起来”的，是模型发现和加载这一步。

这篇文档会围绕下面这条主线展开：

- 目录里有什么模型
- 模型是什么类型
- 该走什么加载逻辑
- 它如何变成 `LoadedModel`
- 它如何被 scheduler 和 transformer 访问

---

## 一、为什么“模型发现与加载”如此重要

在 mlx-serve 里，模型并不是简单地从磁盘路径直接读入内存就结束了。

真正要解决的是几个最核心的问题：

1. 这个路径到底是不是一个模型
2. 它是文本模型还是 media 模型
3. 它是 MLX 原生模型还是 GGUF
4. 它的架构是什么（Gemma / Qwen / LFM2 / Hunyuan / etc.）
5. 它的权重布局和配置是否兼容
6. 需要走哪条前向逻辑

如果这一步做错，后面的 server、scheduler、transformer 都会建立在错误的模型抽象上。

所以模型发现不是“工具层”，而是整个项目的“入口分发层”。

---

## 二、核心入口：model_discovery.zig

最关键的文件是：

- [src/model_discovery.zig](../src/model_discovery.zig)

它的职责可以概括为：

- 扫描目录
- 识别 model 类型
- 识别是否为 GGUF
- 识别是否为 media 模型
- 给出一个统一的 model identity
- 让后续加载器知道要走哪条分支

### 2.1 它干的不是“算模型”，而是“识别模型”

这是最重要的理解。

很多人会误把这层看成普通的文件系统扫描，其实它是整个框架里最关键的判断器之一。

### 2.2 它决定了后续行为

一旦 discovery 判定完：

- 进入常规文本模型路径
- 或者进入 GGUF embedded engine 路径
- 或者进入 media generation 路径

后面所有的代码都建立在这一步的判断之上。

---

## 三、模型类型识别：为什么这一步这么复杂

这个项目支持非常多类型的模型：

- Gemma 系列
- Qwen 系列
- Llama / Mistral
- DeepSeek V4
- LFM2 / Nemotron H
- Laguna
- DiffusionGemma
- media 模型（图像 / 音频 / 视频 / 3D）
- GGUF 模型

每种模型的权重布局、配置字段、视野结构、加载路径都不完全一样。

因此 discovery 需要具备：

- `config.json` 解析能力
- 检查 `model_type`
- 识别权重前缀和布局方式
- 特判 vision / embedding / media model
- 处理 `.gguf` 的 special case

---

## 四、GGUF 逻辑：一个特别关键的分支

文档里反复强调，GGUF 不是普通文本模型的一个小分支，而是一个重要的独立分支。

### 4.1 为什么 GGUF 要单独处理

因为 GGUF 并不使用标准 safetensors 的路径：

- 它通常走 embedded engine
- 比如 llama.cpp
- 或 ds4 这类专用 engine

所以 discovery 会在这里分叉：

- normal MLX model
- GGUF model

### 4.2 它的判定逻辑非常重要

若发现 `.gguf` 文件，通常说明：

- 不走正常的 `model.zig` 结构化加载
- 而要走 embed engine 的路由逻辑
- 之后再由 server / scheduler 以更统一的方式接入

这也是这个项目很独特的一点：

> 不是“所有模型都走同一个前向实现”，而是“模型类型决定了 engine 路径”。

---

## 五、media 模型：另一条很长的分支

媒体生成模型也是 discovery 里一个关键路由点：

- image
- audio
- video
- 3D

这些模型并不是普通 chat 模型的变种，它们走的是：

- [src/gen.zig](../src/gen.zig)
- [src/krea.zig](../src/krea.zig)
- [src/ltx_video.zig](../src/ltx_video.zig)
- [src/minimax_h3.zig](../src/minimax_h3.zig)
- [src/hunyuan3d.zig](../src/hunyuan3d.zig)

### 5.1 这类模型也要先“被识别”，再“被加载”

它们不是因为后缀是 image/audio 就自动能加进系统，而是要满足：

- `model_discovery.isMediaModelType`
- `gen.modalityFromType`
- `peekModelType`

这些逻辑都需要对齐，否则就容易出现：

- 模型被当成普通文本模型
- media generation 端点识别错
- 能力列表和实际加载逻辑不一致

### 5.2 这也是项目里反复强调“保持同步”的地方

文档里明确提到：

> `model_discovery.isMediaModelType` 和 `gen.modalityFromType` 是文档里的重复代码，必须保持同步。

这说明媒体模型不是容易改的新增功能，而是要同时更新多个判定入口，以保持一致。

---

## 六、模型加载真正发生在哪

真正执行“把 parsed model 变成 runtime object”的位置在：

- [src/model.zig](../src/model.zig)
- [src/transformer.zig](../src/transformer.zig)

### 6.1 model.zig：读取和组装配置

这部分负责：

- 解析配置信息
- 解析权重布局
- 构造模型状态
- 记录 tokenizer 等元数据

它并不直接算前向，只负责把模型对象“准备好”。

### 6.2 transformer.zig：架构绑定

这个文件把“模型配置”和“计算路径”绑定起来。

举例来说：

- 某个模型是不是 attention 结构
- 是不是 MoE
- 是不是 hybrid
- 是不是 qwen3_5 / laguna / lfm2
- 是不是 vision tower / M-RoPE / patch embed

这些都在这里决定。

### 6.3 最终产物：LoadedModel

最终，模型加载完成后，系统拿到的是一个可以被 scheduler 使用的对象，例如：

- 生成器对象
- tokenizer
- 缓存对象
- 权重对象
- backend / engine 实例

这就是项目里“运行时模型”的核心抽象。

---

## 七、为什么模型加载还要考虑 prefix cache / memory / residency

模型加载并不会只是“权重进内存”，它还要考虑：

- memory preflight
- dedup / shared model
- prefix cache
- max resident memory
- hot cache / disk cache
- model eviction

这部分在其他文件中也会出现，比如：

- [src/kv_disk_cache.zig](../src/kv_disk_cache.zig)
- [src/prefix_cache.zig](../src/prefix_cache.zig)
- [src/mem_pressure.zig](../src/mem_pressure.zig)

### 关键点

它不是单纯“模型代入运行”，而是：

- 这个模型能不能 load
- load 之后能不能继续推理
- 需要多少内存
- 是否要先 evict 旧模型
- 是否是 hot cache 复用场景

所以模型加载这一步几乎就是“运行前决策中心”。

---

## 八、模型加载不是一次性动作，而是运行时状态入口

很多人会以为“模型一旦 load 成功，就是一直在内存里”。

但这个项目并不是这么简单：

- 模型可能按需加载
- 模型可能按需 unload
- 模型可能被允许 LRU 复用
- 模型可能在主进程里被共享

也就是说，整个系统的模型状态经常是：

- unloaded
- loading
- ready
- evicting
- reloading

这体现了它其实是一个“serve runtime”，而不只是一个单次运行脚本。

---

## 九、从模型发现到 scheduler 的关系

真正的调用链可以理解为：

```text
main/cli
   ↓
模型目录扫描 / discovery
   ↓
识别 model_type / GGUF / media
   ↓
model.zig 组装 runtime model
   ↓
transformer.zig 配置架构
   ↓
scheduler 负责请求接入和运行
```

也就是说：

- discovery 负责“辨认模型是什么”
- model/transformer 负责“把模型变成可以跑的对象”
- scheduler 负责“把请求交给这个对象跑”

这三者是一个完整链条。

---

## 十、最容易忽略的几个点

### 1. discovery 是逻辑中心，不只是扫描工具

很多 bug 都会先在 discovery 这里出现：

- 识别错模型类型
- 走错 engine
- 误判 media model
- 误判 GGUF

### 2. model.zig 不是直接使用模型，而是构造运行时结构

它的工作重点是“建模”，不是“生成 token”。

### 3. transformer.zig 是架构入口，不是最后一步

它是后续 generate/scheduler 的前提，也是整个模型运行逻辑的入口。

### 4. media 模型和 chat 模型同样要经过 discovery

这点很容易忽略，因为 media 模型往往更偏特定生成代码。

---

## 十一、最适合的阅读路径

如果你正在读源码，建议把模型发现和加载这条链视为最优先的入口：

1. [src/main.zig](../src/main.zig)
2. [src/cli.zig](../src/cli.zig)
3. [src/model_discovery.zig](../src/model_discovery.zig)
4. [src/model.zig](../src/model.zig)
5. [src/transformer.zig](../src/transformer.zig)
6. [src/scheduler.zig](../src/scheduler.zig)

这样读最稳定，因为你能先理解：

- 什么模型被找到
- 它为什么会被分到这条路径
- 它为什么能被 scheduler 接管

---

## 十二、结论

模型发现与加载是整个项目中最关键的“分叉入口”。

它决定了：

- 这是哪一类模型
- 走哪条架构逻辑
- 走哪条 engine 路径
- 能不能被 scheduler 正确接管
- 后续 generate 和 server 是否基于正确的 runtime 结构运作

如果你理解了这一层，就已经理解了项目的正确起点。

后面的 scheduler、transformer、generate 都是在这个基础之上展开的。
