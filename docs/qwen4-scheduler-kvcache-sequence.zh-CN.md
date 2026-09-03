# Qwen4 + Scheduler + KV Cache 的源码时序图版（中文）

这篇文档把前面几篇架构总结进一步落成一张“运行时顺序图”，重点回答：

- 一个请求是如何进入 scheduler 的
- slot 是怎样绑定 KV cache 的
- prefill 和 decode 在什么时刻切换
- Qwen4 主干前向在什么位置执行
- speculative / verify / rollback 是如何插进来的

核心 read path：

- [../src/scheduler.zig](../src/scheduler.zig)
- [../src/generate.zig](../src/generate.zig)
- [../src/qwen4_exp.zig](../src/qwen4_exp.zig)
- [../src/transformer.zig](../src/transformer.zig)
- [../src/mtp.zig](../src/mtp.zig)
- [../src/dflash.zig](../src/dflash.zig)

---

## 一、总时序：请求 → slot → KV cache → Qwen4 → decode

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client / HTTP
    participant S as Scheduler
    participant Slot as Slot
    participant KV as KV Cache
    participant G as Generator
    participant T as Transformer
    participant Q as Qwen4Trunk
    participant V as Verify / Spec
    participant C as Commit / Rollback

    Client->>S: 发送生成请求
    S->>Slot: submit() 创建 slot
    Slot->>KV: 分配/绑定 KV cache
    Slot->>G: 构造 ForwardCtx / per-request state
    S->>G: runPrefill(slot)

    G->>T: Generator.initWithOptions(ctx)
    T->>Q: forwardWith(ctx, token_ids)
    Q->>Q: 处理 prompt
    Q->>Q: QSA + Hyper-connection + MoE/GDN trunk
    Q-->>T: 返回 logits / hidden states
    T-->>G: 生成下一 token 提示

    G->>V: 启动 speculative draft / verify

    alt MTP / DFlash / Drafter / PLD 可用
        V->>V: 生成候选 token(s)
        V->>C: 校验 accepted / partial / reject
        alt accepted
            C->>KV: 提交新 cache 位置
            C-->>G: 继续下一轮 decode
        else partial accept
            C->>KV: 只回滚到正确前缀
            C-->>G: 继续从安全状态推进
        else reject
            C->>KV: 清理 draft 产生的 state
            C-->>G: 回退到主干 decode
        end
    else 仅主干 decode
        V-->>G: 直接主干采样下一 token
        G->>KV: 写入新 token 的 kv 状态
    end

    G-->>Client: 流式输出 / 最终输出
```

---

## 二、对应源码层次的语义分解

这张图并不是“概念图”，而是直接映射到源码层：

```text
Client
  -> Scheduler.submit()
      -> Slot.init()
          -> KVCache.initWithConfigAndHeadDim()
          -> ForwardCtx 绑定当前 slot state
      -> runPrefill(slot)
          -> Generator.initWithOptions(ctx)
          -> Transformer.forwardWith(ctx, prompt_tokens)
              -> forwardQwen4With(...)
                  -> QSA / Hyper Stream / GDN / MoE
          -> slot.state = decoding
      -> Generator.next() / runDecodeTick()
          -> speculative draft / verify / commit / rollback
          -> 更新 KV cache
```

---

## 三、最关键的一段：slot 与 KV cache 的绑定

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant Slot as Slot
    participant Ctx as ForwardCtx
    participant KV as KV Cache
    participant G as Generator

    S->>Slot: submit() 创建 slot
    Slot->>KV: 为请求分配缓存
    Slot->>Ctx: 创建 ctx
    Ctx->>KV: 持有 cache 引用
    Slot->>G: 将 ctx 传入 generator
    G->>Ctx: 读取当前生成状态
    G->>KV: 在每轮 decode 时写入新 token 的 K/V
```

这里非常关键：

- `Slot` 不是只存“请求参数”
- `Slot` 真正持有的是“本轮生成的 live state”
- `KV cache` 是那份 live state 的核心容器
- `ForwardCtx` 是两者之间的桥接对象

这也解释了为什么 Qwen4 runtime 的大部分状态都不是全局单例，而是 slot-scoped。

---

## 四、prefill 与 decode 的时序切换

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant Slot as Slot
    participant G as Generator
    participant T as Transformer
    participant Q as Qwen4

    S->>Slot: 进入 pending slot
    S->>G: runPrefill(slot)
    G->>T: 取 prompt token
    T->>Q: 主干前向计算 prompt
    Q-->>T: hidden states / logits
    T-->>G: 完成 prefill
    G-->>Slot: 标记 state = decoding

    loop decode token-by-token
        G->>T: next token 推理
        T->>Q: 当前 token 的 Qwen4 forward
        Q-->>T: logits / next-token distribution
        T-->>G: 返回采样结果
        G->>Slot: 更新 cache / step / generated len
    end
```

这里的状态转换非常重要：

```text
pending_prefill -> decoding
```

也就是说，真正的“模型开始生成 token”不在 request 进入的一开始，而是在 prefill 已经把上下文状态装好之后。

---

## 五、speculative draft 和 verify 的插入时序

```mermaid
sequenceDiagram
    autonumber
    participant G as Generator
    participant Spec as Spec Draft
    participant V as Verify
    participant KV as KV Cache
    participant Roll as Rollback

    G->>Spec: 触发 speculative draft
    Spec->>Spec: 根据当前 context 生成候选 token(s)
    Spec->>V: 递交候选序列

    V->>V: 进行 verify / accept 判断
    alt 全部接受
        V->>KV: 提交整段 cache
        V-->>G: 返回 accepted
    else 部分接受
        V->>Roll: 回滚到正确前缀
        Roll->>KV: 只保留有效前缀
        Roll-->>G: 继续 decode
    else 全部 reject
        V->>KV: 清理 draft state
        V-->>G: 回退主干 decode
    end
```

从中可以看出：

- spec draft 并不是独立输出最终答案
- 它只是给主干提供一个“候选未来序列”
- 真正决定生成是否正确的，是 verify + rollback 这条线

---

## 六、把它压成一句最短描述

```text
Client -> Scheduler.submit -> Slot.init -> KVCache.bind -> runPrefill -> Generator.init -> Transformer.forward -> Qwen4Trunk -> decode loop -> speculative verify -> commit or rollback -> next token
```

---

## 七、总结

Qwen4 + Scheduler + KV Cache 的真正运行时序，不是“单模型直接输出 token”，而是：

1. 请求进入 scheduler
2. 创建 slot 并绑定 KV cache
3. prefill 完成上下文状态建立
4. Qwen4 主干开始前向
5. decode loop 逐 token 生成
6. speculative draft / verify 在中间插入
7. 成功则 commit，失败则 rollback
8. 继续推进下一轮生成

这条时序也正是 mlx-serve 里 Qwen4 runtime 的核心骨架。
