# Qwen4 + Scheduler + KV Cache 的源码级函数调用链（中文）

这篇文档把 Qwen4 的运行时链路压到“源码级函数调用顺序”上，目的不是讲概念，而是把调用关系串起来：

- 请求从哪里进入 scheduler
- slot 是如何建出来的
- KV cache 是如何分配和绑定的
- prefill 之后 Generator 是怎么接上的
- Qwen4 main forward 是在哪里触发的
- decode tick 与 rollback 发生在什么时刻

相关源码：

- [../src/scheduler.zig](../src/scheduler.zig)
- [../src/generate.zig](../src/generate.zig)
- [../src/qwen4_exp.zig](../src/qwen4_exp.zig)
- [../src/transformer.zig](../src/transformer.zig)

---

## 一、最核心的调用链：从 request 到 decode

```text
Scheduler.submit()
  -> Slot.init()
      -> KVCache.initWithConfigAndHeadDim()
      -> SSMCacheEntry init (if needed)
      -> ForwardCtx init
  -> Scheduler.pending append
  -> inference thread drains pending
  -> runPrefill(sch, slot)
      -> Generator.initWithOptions(.{ .ctx = slot.ctx, ... })
      -> transformer.forwardWith(...)
          -> forwardQwen4With(...)
              -> QSA sparse attention
              -> Hyper-connection residual stream
              -> main trunk forward
      -> slot.state = decoding
      -> Generator.next() / runDecodeTick()
```

这条链路是最重要的主线：

> 请求进入 scheduler → slot 和 KV cache 建立 → prefill 产出可用状态 → Qwen4 主干前向执行 → decode tick 继续生成。

---

## 二、入口：Scheduler.submit()

真正的入口在：

- [../src/scheduler.zig](../src/scheduler.zig): `Scheduler.submit()`

它的职责是：

1. 读取 per-request 配置
2. 取模型配置
3. 创建 `Slot`
4. 绑定 `KVCache`
5. 把 slot 放进 scheduler 的 pending 队列
6. 等待 inference thread 处理

对应源码位置：

- `src/scheduler.zig` 中的 `submit(self: *Scheduler, params: SubmitParams) !*Slot`

它最关键的动作：

- `const slot = try Slot.init(...)`
- `self.pending.append(self.allocator, slot)`

所以，`submit` 实际上是“请求进入调度器”的总入口。

---

## 三、Slot.init(): KV Cache 和状态的真正落地

随后走的是：

- [../src/scheduler.zig](../src/scheduler.zig): `Slot.init()`

这个函数里，最关键的事情是：

1. 为当前请求分配 `KVCache`
2. 初始化 `SSMCacheEntry`（如果模型需要 hybrid/SSM 状态）
3. 绑定 `ForwardCtx`
4. 记录 prompt / sampling / EOS / spec flags
5. 设定 slot 状态为 `pending_prefill`

对应的核心函数链：

```text
Slot.init()
  -> KVCache.initWithConfigAndHeadDim()
  -> maybe allocate SSM cache
  -> slot.ctx = ForwardCtx{ .cache = &slot.cache, ... }
```

这一步非常关键，因为后面所有 decode 和 spec draft 都依赖这个 slot 上的 `ctx`。

也就是说：

> 一切真正的 per-request state，都是在 `Slot.init()` 里“绑定到 slot”而不是直接挂在全局 transformer 上。

---

## 四、KV Cache 的绑定位置：ForwardCtx

`ForwardCtx` 是整个运行时里非常关键的桥接对象：

- 它持有 `cache`
- 持有 `moe_seq_offset`
- 持有 `ssm_entries`
- 持有 `vision_embeddings`
- 持有 `mrope` 相关信息

它在 `Slot.init()` 中写成：

```text
slot.ctx = .{
    .cache = &slot.cache,
    .moe_seq_offset = &slot.moe_seq_offset,
    .ssm_entries = slot.ssm_entries,
    .vision_embeddings = slot.vision_embeddings,
    .mrope_pos = slot.mrope_pos,
    ...
};
```

之后，`runPrefill()` 会把它传给 `Generator.initWithOptions()`：

```text
runPrefill(sch, slot)
  -> Generator.initWithOptions(.{ .ctx = slot.ctx, ... })
```

所以，从结构上看：

```text
Slot -> ForwardCtx -> Generator -> Transformer.forwardWith -> Qwen4
```

这条链路是最关键的 state binding 链。

---

## 五、prefill 入口：runPrefill()

真正的 prefill 启动入口在：

- [../src/scheduler.zig](../src/scheduler.zig): `runPrefill()`

它的核心顺序是：

```text
runPrefill(sch, slot)
  -> refresh slot.ctx
  -> decide spec wiring (mtp / drafter / dflash / pld)
  -> Generator.initWithOptions(...)
  -> generator starts to consume prompt
  -> slot.state = decoding
```

这里有几个关键点：

- `slot.ctx.cache = &slot.cache`
- `slot.ctx.ssm_entries = slot.ssm_entries`
- spec 路径被在此处装配：`specInitWiring(...)`

也就是说，prefill 前后的状态切换是：

```text
pending_prefill -> decoding
```

并且在这一步里，`Generator` 已经拿到了正确的 `ctx`，可以进入真正的 token 生成。

---

## 六、Generator.initWithOptions(): 让 Qwen4 真正开始工作

`Generator` 的初始化在：

- [../src/generate.zig](../src/generate.zig): `Generator.initWithOptions(...)`

它负责：

- 把 `ctx` 和 transformer 绑定
- 设置 `next_token_id`
- 初始化 `sampling`
- 建立 prompt token / generated token 相关状态
- 置入 `spec` 相关开关（PLD / drafter / MTP / DFlash）

这里的关键链：

```text
Generator.initWithOptions()
  -> self.xfm = transformer
  -> self.ctx = ctx
  -> self.next_token_id = first token
  -> setup spec states
```

这是 decode 系统真正“开机”的位置。

---

## 七、实际的 Qwen4 forward：Transformer.forwardWith + forwardQwen4With

Qwen4 的真正前向从 transformer 这里进入：

- [../src/transformer.zig](../src/transformer.zig): `Transformer.forwardWith(...)`
- [../src/qwen4_exp.zig](../src/qwen4_exp.zig): `forwardQwen4With(...)`

对应的调用链是：

```text
Generator.next() / lazyForward(...)
  -> xfm.forwardWith(ctx, token_ids)
      -> Transformer.forwardQwen4With(ctx, token_ids)
          -> hyper-connection read/write
          -> QSA attention mask / sparse attention
          -> GDN / MoE trunk forward
          -> logits output
```

最关键的一点：

> `ctx` 里拿到的 `cache`、`ssm_entries`、`moe_seq_offset` 这些对象，才是 Qwen4 前向真正要写入的 per-request state。

这也是为什么 `slot.ctx` 在整个链路里如此关键：

- 它不是可有可无配置
- 它是生成状态的真实 live state

---

## 八、decode tick：Generator.next() 与 runDecodeTick()

真正逐 token 生成的逻辑在：

- [../src/generate.zig](../src/generate.zig): `Generator.next()`
- [../src/scheduler.zig](../src/scheduler.zig): `runDecodeTick()`

典型链路：

```text
runDecodeTick(sch, active_slots)
  -> for each slot
      -> slot.legacy_gen.next()
          -> lazyForward(...)
          -> sample token
          -> update generated_ids
          -> update KV cache
          -> maybe spec accept/rollback path
```

这是 decode 阶段的真正在跑的函数链。

注意：

- `runDecodeTick` 是 scheduler 层在调度 decode 轮询
- `Generator.next` 是 per-slot token 生成
- `Transformer` 真正算出 logits

所以结构上是：

```text
Scheduler.runDecodeTick
  -> per-slot Generator.next
      -> Transformer.forwardWith
          -> Qwen4 forward path
```

---

## 九、speculative draft 和 rollback 的插入点

Speculative 路径并不在 scheduler 的最外层“独立绕开”模型，而是嵌入在 `Generator.next()` 的内部逻辑中。

在 [../src/generate.zig](../src/generate.zig) 里，相关逻辑包括：

- `nextPld()`
- `nextDrafter()`
- `nextDflash()`
- `nextMtp()`

它们的通用顺序：

```text
Generator.next()
  -> maybe nextPld / nextDrafter / nextDflash / nextMtp
      -> draft candidate(s)
      -> verify candidate(s)
      -> accept / partial accept / reject
      -> commit or rollback cache state
```

这里的关键点：

- speculative draft 不是另起一套“完全不同状态机”
- 它是 Generator 内部对当前 `ctx.cache` 做“临时推进 + 验证 + commit/rollback”的过程

所以从源码角度讲：

```text
Generator.next
  -> draft candidate
  -> verify
  -> cache commit or cache truncate
```

---

## 十、从状态机角度的 backbone 调用链

你可以把整个真实调用链压成下面这个版本：

```text
Scheduler.submit
  -> Slot.init
      -> KVCache.initWithConfigAndHeadDim
      -> ForwardCtx.init
  -> runPrefill
      -> Generator.initWithOptions
          -> Transformer.forwardWith
              -> forwardQwen4With
                  -> QSA + hyper + trunk
      -> slot.state = decoding
  -> runDecodeTick
      -> Generator.next
          -> lazyForward / sample token
          -> update slot.cache
          -> spec draft verify / rollback
          -> continue next token
```

这是最贴近源码的“主链”。

---

## 十一、最短记忆版

如果你只想保留最关键的一条链：

```text
submit -> Slot.init -> KVCache + ForwardCtx -> runPrefill -> Generator.initWithOptions -> Transformer.forwardWith -> forwardQwen4With -> Generator.next -> runDecodeTick -> commit / rollback -> next token
```

---

## 十二、继续阅读的推荐顺序

建议按这个顺序继续往下看：

1. [../src/scheduler.zig](../src/scheduler.zig)
2. [../src/generate.zig](../src/generate.zig)
3. [../src/transformer.zig](../src/transformer.zig)
4. [../src/qwen4_exp.zig](../src/qwen4_exp.zig)

这样你能看清：

- 调度层在哪里创建状态
- cache 在什么时机被写
- decode 是如何从 generator 接到 transformer
- Qwen4 的主前向在哪里真正执行

---

## 十三、总结

最简洁的结论是：

> Qwen4 的源码级运行链路，本质上是 “scheduler 创建 slot → slot 持有 KV cache → prefill 构造 generator/context → transformer 执行 Qwen4 forward → decode loop 继续推进 → spec verify / rollback 驱动 cache 更新”。

这条链路就是整个 mlx-serve Qwen4 runtime 的真实骨架。
