# Qwen4 源码函数名对照表（最终版）

这篇文档把前面所有“架构图 / 总图 / 时序图”压成一个最实用的表格版：

- 哪个文件定义了这个函数
- 这个函数主要负责什么
- 它在 Qwen4 运行时中的位置是什么
- 它和前后两个环节的关系是什么

目标：把源码阅读从“看图”直接变成“按表查函数”。

---

## 一、核心函数对照表

| 层次 | 文件 | 函数名 | 作用 | 在运行时中的位置 |
|---|---|---|---|---|
| Scheduler | src/scheduler.zig | `Scheduler.submit` | 请求进入 scheduler 的入口 | 首个入口，创建 slot 并投递运行 |
| Scheduler | src/scheduler.zig | `Slot.init` | 创建 per-request live state | 请求有独立生成状态的落点 |
| Scheduler | src/scheduler.zig | `runPrefill` | prompt prefill 阶段入口 | 把上下文状态装进生成器 |
| Scheduler | src/scheduler.zig | `runDecodeTick` | decode tick 调度入口 | token 逐步生成的调度循环 |
| KV Cache | src/scheduler.zig / src/transformer.zig | `KVCache.initWithConfigAndHeadDim` | 分配并初始化缓存 | 真正持有历史 K/V 的容器 |
| Context | src/scheduler.zig | `ForwardCtx` | 绑定 cache + seq state + 生成上下文 | 连接 slot、generator、transformer |
| Generator | src/generate.zig | `Generator.initWithOptions` | 绑定 transform + ctx + sampling 设定 | 生成器开始工作的初始化点 |
| Generator | src/generate.zig | `Generator.next` | 生成下一个 token 的核心逻辑 | decode 主循环 |
| Transformer | src/transformer.zig | `Transformer.forwardWith` | 通用前向入口 | 把 token 状态交给模型前向 |
| Qwen4 | src/qwen4_exp.zig | `forwardQwen4With` | Qwen4 主干前向实现 | 模型真正计算 logits 的地方 |
| Qwen4 | src/qwen4_exp.zig | QSA 相关逻辑 | 稀疏注意力/索引 mask | 在 Qwen4 分支中负责 attention 的稀疏化 |
| Qwen4 | src/qwen4_exp.zig | Hyper-connection 逻辑 | 附加残差状态流 | 在 trunk 中参与额外信息流 |
| Qwen4 | src/qwen4_exp.zig | GDN / MoE 逻辑 | 主干计算的关键 path | 形成 token 表征/上下文状态 |
| Spec | src/mtp.zig | `nextMtp` | MTP speculative draft | 生成候选 future token |
| Spec | src/dflash.zig | `nextDflash` | DFlash block draft | 在 block 级别进行猜测 |
| Spec | src/drafter.zig | `nextDrafter` | drafter 协助 draft | 作为通用 draft 路径 |
| Spec | src/pld_index.zig | `nextPld` | prefix/历史文本匹配 draft | 通过重用历史内容进行 draft |
| Verify | src/generate.zig / src/scheduler.zig | verify / accept 判断 | 判定候选 token 是否可接收 | 指定是否提交到 cache |
| Recovery | src/generate.zig / src/scheduler.zig | rollback | 回滚到有效前缀 | 处理 partial accept / reject |
| Commit | src/generate.zig / src/scheduler.zig | commit cache update | 提交正确的 K/V 状态 | 保证下一个 step 正确推进 |

---

## 二、最重要的调用链（按顺序）

```text
Scheduler.submit
  -> Slot.init
      -> KVCache.initWithConfigAndHeadDim
      -> ForwardCtx init
  -> runPrefill
      -> Generator.initWithOptions
      -> Transformer.forwardWith
          -> forwardQwen4With
              -> QSA
              -> Hyper-connection
              -> GDN / MoE
              -> logits
  -> Generator.next
      -> runDecodeTick
      -> nextMtp / nextDflash / nextDrafter / nextPld
      -> verify / accept
      -> rollback or commit
```

---

## 三、按“看代码”最实用的阅读顺序

### 第一层：入口

1. `Scheduler.submit`
2. `Slot.init`
3. `KVCache.initWithConfigAndHeadDim`

### 第二层：状态绑定

4. `ForwardCtx`
5. `Generator.initWithOptions`

### 第三层：主干推理

6. `Transformer.forwardWith`
7. `forwardQwen4With`
8. QSA / hyper / GDN / MoE / logits

### 第四层：生成循环

9. `Generator.next`
10. `runDecodeTick`

### 第五层：speculative / 验证

11. `nextMtp`
12. `nextDflash`
13. `nextDrafter`
14. `nextPld`
15. verify / accept / rollback / commit

---

## 四、最简记忆版

```text
submit -> slot -> kvcache -> ctx -> generator -> transformer -> qwen4 -> next -> spec -> verify -> commit/rollback
```

---

## 五、终局结论

如果说一句最实用的话：

> 直接按这个表看源码，最核心的主线就是：请求进入 `Scheduler`，在 `Slot` 中绑定 `KVCache` 和 `ForwardCtx`，然后 `Generator` 调用 `Transformer.forwardWith`，最终落到 `forwardQwen4With` 做 Qwen4 主干计算；之后 `Generator.next` + `runDecodeTick` 进入 decode 循环，并在 `nextMtp / nextDflash / nextDrafter / nextPld` 之上插入 `verify` 和 `rollback`。

这就是 Qwen4 + Scheduler + KV Cache 的源码函数名总脉络。
