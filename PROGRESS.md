# Qwen4-Exp MoE 专家流式加载（slotstream 移植）— 进度

> 本轮把 `expert_store.zig` 从上一版的**错误设计**（每层一个 bf16 池、从"已常驻内存的
> `experts.{i}` 权重 memcpy"、自带 forward）**整体重写**为 `expert-streaming-handoff.md`
> 描述、并被 slotstream 实测验证过的正确设计：**全局共享的量化槽池 + 磁盘 pread + 复用
> gather_qmm 内核 + 专家→槽位重映射**。上一版从未编译通过、也与真实 `MoeMlpWeights` 的
> 量化 `switch_*` 布局对不上（详见"关键修正"）。

## 编译/测试状态（全部本机实测通过）

- `zig build -Doptimize=ReleaseFast`：**通过**（改到 main.zig，走完整构建）
- `zig build test`：**通过**，含本轮新增 6 个纯逻辑单测
- 工具链：`.zig-toolchain/zig` 0.17.0-dev；真实检查点几何已 `python` 核对（gs32、record=3,072,000 B、49 套 switch_mlp=48 主干+1 MTP）

## 已完成（有磁盘变更，全部编译通过）

| 文件 | 内容 |
|---|---|
| `src/expert_store.zig` | **重写**：`TensorRef`/`ExpertStore`(解析分片头→按 layer×piece 定位 `model.layers.{d}.mlp.switch_mlp.*`，`pread`+`F_NOCACHE`) + `SlotPool`(9 个量化池数组 + CLOCK `map/key_of/ref/pin/hand` + `ensure`(去重→victim→readPiece→`mlx_scatter`→换句柄) + `unpinAll`/`resize`(冷重建+`mlx_clear_cache`) + 统计 + `POOL_FLOOR` 常量 + `isTrunkSwitchKey`/`parseTrunkKey` + 6 单测 |
| `src/mlx.zig` | 新增 `extern mlx_scatter`（原缺失） |
| `src/model.zig` | `ModelConfig` 加 `expert_stream/expert_pool_gb/experts_per_layer`；进程级 `expert_stream_requested/expert_pool_gb_req/experts_per_layer_req` + 加载门 `g_expert_stream`；`parseConfig` 仅对 qwen4 提升请求；`shouldKeepWeightKey` 流式时丢主干 `switch_mlp`（放行 `mtp`） |
| `src/transformer.zig` | `MoeMlpWeights.expert_layer_id`（免改 7 处调用点）；通用绑定分支流式时 `switch_*` 绑 null + 设 layer_id；**移除每层 `expert_pool` → 全局 `Transformer.expert_pool`/`expert_store_ptr`** + `deinit`；`loadQwen4Mtp` 置 `mcfg.expert_stream=false`；`moeMLP2` 内 `mwe` 池块替换（判别 `switch_gate_w.ctx==null`）+ 路由后**argsort 之前**专家→槽位重映射 + 层末 `unpinAll` defer；`init` qwen4 分支批量 eval **之后**构造池 + 预算钳制 + `[expert] pool` 日志 |
| `src/main.zig` | `--expert-stream/--expert-pool-gb/--experts-per-layer` + `MLX_SERVE_EXPERT_STREAM`；把看门狗水位镜像进 `expert_store.g_watermark/hysteresis`；加载前后设/清 `g_expert_stream`；`--help` |
| `src/scheduler.zig` | `expertStreamResidentBytes` 纯函数+单测；preflight/`bytes_resident` 流式按"非专家+池"计费；`PoolResizeRequest`+`pool_resize_queue`+`runPoolResize`(推理线程)+`expertPoolShrinkTarget`/`shrinkExpertPool`；tick 顶部先缩池再加载；idle-wait 谓词纳入新队列 |
| `src/server.zig` | 看门狗 `.evict` 先投缩池、再退模型（解压阀走单推理线程，第二线程不碰 mlx） |
| `tests/test_expert_streaming.sh` | 运行时验收 harness（golden equivalence 30 vs 181 专家/层、`[expert] pool` 行、吞吐下限哨兵、探针压力互锁），**需真机 + 你亲手起服务** |
| `src/tests.zig` | 显式 `@import("expert_store.zig")` |

## 关键修正（相对上一版错误设计）

1. **池是全局共享的**，不是每层一个（slotstream 实测：CLOCK 让热层向冷层借槽）。
2. **池存量化 9 块**（U32 权重 + BF16 scales/biases，g=32），不是反量化 bf16——反量化正是 77GB 常驻的元凶。
3. **从磁盘 `pread`**，不是从"已常驻权重 memcpy"——后者根本不省内存，与目标相悖。
4. **复用 `gather_qmm`/`gatherQmv`/take 三条内核**，只把 `rhs_indices` 换成槽位；MLX 内核一行不改。上一版自造 `forward()` 是错路。
5. **重映射必须在 argsort 之前**：`gather_qmm` 快路径假定 `rhs_indices` 单调，事后改值会**静默算错**（无报错）。
6. **`switch_*` 流式绑 null**（不是绑池块）：绑池块会多一层引用、破坏 buffer 捐赠→每次 ensure 退化成整池拷贝（16GB≈60ms，decode 废）；绑 null 还能让加载末尾的 `appendHybridMlpWeights` 自动跳过专家。
7. 真实 `MoeMlpWeights` 是 `switch_gate_w/s/b…` 量化布局，**根本没有** `experts.{i}.gate_w` 三元组——上一版的前提不存在，且它用 `mlx_array_new_data(&pool,null,dtype,shape,len)`（5 参）调用，签名对不上，从没编译过。

## 待你在真机验证（Step 6 运行时项，AI 不代跑高影响操作）

```
# 1) 冒烟 + [expert] pool 日志 + 不 OOM
./zig-out/bin/mlx-serve serve --model ~/.mlx-serve/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit \
    --port 8001 --expert-stream --expert-pool-gb 16
#   期望日志：[expert] pool 5208 slots (…GiB), 48 layers, 3.072MB/record, …
#   MLX 常驻 ≈ 非专家(~36GB safetensors 非专家部分) + 16GB 池，不 OOM

# 2) golden equivalence（30 vs 181 专家/层，贪心输出逐字节一致）+ 吞吐哨兵
BIN=./zig-out/bin/mlx-serve PORT=8001 bash tests/test_expert_streaming.sh

# 3) 压力互锁（不制造真实内存压力）：MLX_SERVE_MEM_PROBE_FILE 注入
#    期望：低压先 "shrinking expert slot pool to …"，不先 "evicting idle resident model"；
#    缩到下限仍低压 30s 才 exit(0)
```

## 已知边界 / 后续可做

- **真机首轮崩溃（已修）**：首帧前向 `mlx_scatter` 报 `Updates with 3 dimensions does not match the sum of the array (3) and indices (1) dimensions`。MLX 通用 scatter 要求 `updates.ndim == a.ndim + indices.ndim`，对一维 `[nm]` 索引即 `[nm,1,d1,d2]`（正是 Python `a[idx]=v` 底层 `mlx_scatter_args_array` 插入的 singleton 轴）。`ensure()` 已把 staging 重塑为 4 维（连续、纯形状重标），ReleaseFast 重建通过、`zig build test` 绿。
- `readPiece` 现为**串行 pread**（v1，handoff/崩溃日志认定的可接受档：QD1≈9.5GB/s，仅比 QD8 慢 ~1.8×，零线程复杂度）。`MLX_SERVE_EXPERT_IO_QD` 并行化留作性能跟进。
- 流式仅覆盖主干 48 层；MTP 草稿层走常驻（`mcfg.expert_stream=false`），其 1.57GB 惰性物化，已接受。
- **捐赠退化哨兵现可直接观测**：`MLX_SERVE_EXPERT_STATS=N` 每 N 次流式层访问打印 `hit=… per_fill=…us`。`SlotPool` 已带 `fill_ops/io_bytes/fill_ns` 计数。per_fill 从几十 µs 跳到毫秒级＝buffer 捐赠退化成整池拷贝（golden equivalence 抓不到"只变慢不变错"，这个能抓）。

## 文件清单

| 文件 | 状态 |
|---|---|
| `src/expert_store.zig` | 重写（新设计），git 有变更 |
| `src/{model,transformer,main,scheduler,server,mlx,tests}.zig` | 接线，git 有变更 |
| `tests/test_expert_streaming.sh` | 新建（运行时验收 harness） |
| `PROGRESS.md` | 本文件（已替换为准确状态） |
| `expert-streaming-handoff.md` | 只读依据，未改 |
