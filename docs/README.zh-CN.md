# mlx-serve 文档中文总览

这份文档是整个 docs 目录的中文入口，按“阅读顺序 + 关注点”给出建议路径。

---

## 一、建议的阅读顺序

### 1. 先看总览

- [source-flow.zh-CN.md](source-flow.zh-CN.md)  
  先理解系统从请求到生成的大体流程。

- [source-flow-deep.zh-CN.md](source-flow-deep.zh-CN.md)  
  进一步看更细的源码运行层次。

### 2. 再看模型和调度

- [model-loading-flow.zh-CN.md](model-loading-flow.zh-CN.md)  
  解释模型发现、加载、配置解析和 runtime state 注入。

- [scheduler-flow.zh-CN.md](scheduler-flow.zh-CN.md)  
  解释 slot、queue、prefill、decode、batch 和 spec 路径如何协作。

- [transformer-generate-flow.zh-CN.md](transformer-generate-flow.zh-CN.md)  
  解释模型前向、生成循环与采样之间的关系。

### 3. 再看 HTTP / API 层

- [server-api-flow.zh-CN.md](server-api-flow.zh-CN.md)  
  从 API 层看 OpenAI / Anthropic / Ollama / SSE / WS / LAN 的接入方式。

### 4. 然后看 Qwen4

- [qwen4-architecture.zh-CN.md](qwen4-architecture.zh-CN.md)  
  总体架构。

- [qwen4-qsa-ngram.zh-CN.md](qwen4-qsa-ngram.zh-CN.md)  
  QSA 与 n-gram / PLE 相关机制。

- [qwen4-hyper-connection.zh-CN.md](qwen4-hyper-connection.zh-CN.md)  
  Hyper-connection 残差流。

- [qwen4-mtp-spec.zh-CN.md](qwen4-mtp-spec.zh-CN.md)  
  MTP speculative draft 机制。

- [dflash-spec-draft.zh-CN.md](dflash-spec-draft.zh-CN.md)  
  DFlash block draft 机制。

- [spec-draft-comparison.zh-CN.md](spec-draft-comparison.zh-CN.md)  
  对比 Drafter / PLD / MTP / DFlash。

### 5. 最后看综合总图

- [qwen4-mtp-dflash-drafter-relationship.zh-CN.md](qwen4-mtp-dflash-drafter-relationship.zh-CN.md)  
  综合关系图。

- [qwen4-mtp-dflash-scheduler-sequence.zh-CN.md](qwen4-mtp-dflash-scheduler-sequence.zh-CN.md)  
  时序图版。

- [qwen4-scheduler-kvcache-rollback-state-machine.zh-CN.md](qwen4-scheduler-kvcache-rollback-state-machine.zh-CN.md)  
  状态机版。

- [qwen4-scheduler-kvcache-callchain.zh-CN.md](qwen4-scheduler-kvcache-callchain.zh-CN.md)  
  源码函数调用链版。

- [qwen4-final-architecture-map.zh-CN.md](qwen4-final-architecture-map.zh-CN.md)  
  一页总图版。

- [qwen4-source-function-map.zh-CN.md](qwen4-source-function-map.zh-CN.md)  
  源码函数名总图版。

- [qwen4-source-function-table.zh-CN.md](qwen4-source-function-table.zh-CN.md)  
  源码函数名对照表版。

---

## 二、适合的阅读方式

### 面向快速理解

按这个顺序：

1. [source-flow.zh-CN.md](source-flow.zh-CN.md)
2. [scheduler-flow.zh-CN.md](scheduler-flow.zh-CN.md)
3. [qwen4-architecture.zh-CN.md](qwen4-architecture.zh-CN.md)
4. [qwen4-final-architecture-map.zh-CN.md](qwen4-final-architecture-map.zh-CN.md)

### 面向源码对照

按这个顺序：

1. [qwen4-source-function-map.zh-CN.md](qwen4-source-function-map.zh-CN.md)
2. [qwen4-source-function-table.zh-CN.md](qwen4-source-function-table.zh-CN.md)
3. [qwen4-scheduler-kvcache-callchain.zh-CN.md](qwen4-scheduler-kvcache-callchain.zh-CN.md)
4. [../src/scheduler.zig](../src/scheduler.zig)
5. [../src/generate.zig](../src/generate.zig)
6. [../src/transformer.zig](../src/transformer.zig)
7. [../src/qwen4_exp.zig](../src/qwen4_exp.zig)

### 面向 spec / draft 细节

按这个顺序：

1. [qwen4-mtp-spec.zh-CN.md](qwen4-mtp-spec.zh-CN.md)
2. [dflash-spec-draft.zh-CN.md](dflash-spec-draft.zh-CN.md)
3. [spec-draft-comparison.zh-CN.md](spec-draft-comparison.zh-CN.md)
4. [qwen4-mtp-dflash-drafter-relationship.zh-CN.md](qwen4-mtp-dflash-drafter-relationship.zh-CN.md)

---

## 三、这套文档的目标

这份 README 的重点不是让你把所有文档都完整读完，而是让你能按“问题切入”来读：

- 想知道系统总体流程：读 source-flow
- 想知道模型如何加载：读 model-loading-flow
- 想知道调度如何工作：读 scheduler-flow
- 想知道 Qwen4 怎么实现：读 qwen4 系列文档
- 想知道源码函数到底怎么串起来：读 callchain / function map / function table

---

## 四、建议的最短路线

如果时间有限，最推荐的最短路径是：

```text
source-flow.zh-CN.md
-> scheduler-flow.zh-CN.md
-> qwen4-architecture.zh-CN.md
-> qwen4-mtp-spec.zh-CN.md
-> qwen4-final-architecture-map.zh-CN.md
-> qwen4-source-function-table.zh-CN.md
```

这条路线最适合建立“整体认知 → 关键细节 → 源码对照”的闭环。

---

## 五、总结

docs 目录整体上分成三层：

1. 总体流程层：source-flow / scheduler-flow / transformer-generate-flow
2. 模型与 spec 层：qwen4 / mtp / dflash / drafter / pld
3. 源码对照层：callchain / function map / function table / final architecture map

如果你是为了理解这个项目的运行时结构，重点看第 1 层 + 第 2 层；如果你是为了继续写代码或定位实现，重点看第 3 层。

---

## 六、结论

这份文档已经整合了整个中文阅读入口，并把 Qwen4 / Scheduler / KV Cache / MTP / DFlash 等关键知识串成一套可读的导航结构。

如果你要继续深入，直接从这里进入最相关的文档即可。
