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

## 性能调优历程（Vontra / 128GB / temp=0 / 256-token 长请求，本机实测）

| 里程碑 | 关键改动 | decode tok/s |
|---|---|---|
| 初接通（有隐性 bug） | 全局槽池 + 重映射 + 散射 | **0.38** |
| `9a44e5f` 捐赠修复 | scatter 后**先释放我方池句柄再 eval**（`is_donatable` 才为真，输出复用池 buffer 不整块拷）→ `copy_per_fill 2.79GB→0` | **3.50** |
| `9a44e5f` QD 读盘 | 持久化 `ReadPool`(8 线程) 并行下发 9×nm 个 pread → `disk 1.53ms→0.39ms` | **8.95** |
| `2861671` 批量 eval | 9 块 scatter 合并为**一次** `mlx_eval`（每 fill 9 次 GPU 同步 barrier→1） | **12.9** |
| 池调优（无代码改动） | `--expert-pool-gb` 16→48（命中 0.82→0.90，miss −42%） | **19.0** |

**当前落点：单流 ~19 tok/s（进入 slotstream 12–20 参照带）；4 路并发聚合 37.5 tok/s。**

`per_fill` 拆分（48GB 稳态，`MLX_SERVE_EXPERT_STATS` 实测）：总 690us = **disk 407us（59%）+ 批量 eval 146us（21%）+ scatter建图/host→dev 138us**。→ 同步已很便宜，剩余大头是真·读盘延迟。

## 实测结论（原"待验证"三项）

- **golden-equivalence（池大小不改数学）**：⚠️ 本构建 **temp=0 贪心本身非确定**——同配置(181/层)冷启动连跑 3 次得到 3 个不同输出（前 ~221 字符一致，在一个近平局 token 上分叉）。跨池(30 vs 181)的分叉位置 = 同配置连跑的分叉位置，说明**发散与池大小无关、非流式 bug**。"逐字节一致"标准在此构建不可达；流式正确性以"确定性前缀完全一致 + 仅在平局处分叉"确认。
- **压力互锁**：预算钳制生效（启动日志 `budget 76.83 GiB`，请求超预算会 clamp+WARN）；缩池解压阀代码就绪。
- **内存峰值**（`wired+compressor+anon−purgeable`，看门狗口径）：16GB 池整机峰值 45.7GB（起服务前基线 ~19.6GB → **服务自身 ≈26GB** = 16GB 池 + 非专家/MTP/KV/暂存）；ngram 表 32GB 是可回收 file-backed 页，不计入。⚠️ 之前报的"RSS 3.5GB"是**进程 RSS、不含 Metal/GPU 分配，严重低估**，作废。

## 池大小 sweet spot / 并发 / 预取决策

- **48GB 是甜点**：命中 0.90；64GB 无增益（hit 0.898、miss 与 48GB 逐位相同——该请求活跃专家集 48GB 已装满）。**建议启动写 `--expert-pool-gb 48`**。
- **并发（系统已有连续批处理）是吞吐正解**：4 路 → 聚合 37.5 tok/s（≈单流 2×），命中升到 **0.961**（多序列共享热专家）。A 读盘时 B/C 的 gather 占 GPU，磁盘等待被天然盖住。
- **跨层预取：经依赖分析 + 并发实测，决定不做**。原因：(1) L+1 专家集依赖 L+1 路由、路由依赖 L 的 MoE 输出（含 L 读盘）→ 硬串行；(2) 单流 batch=1 时 GPU 等 disk 期间无独立活可重叠；(3) hit 已 0.90 能省的读盘本就少。并发场景已由批处理解决。

## 已知边界 / 后续可做

- `ReadPool` worker 数固定 8（QD8）；`MLX_SERVE_EXPERT_IO_QD` 目前未做成可调（`run` 内 `N=8` 常量）。
- 流式仅覆盖主干 48 层；MTP 草稿层走常驻（`mcfg.expert_stream=false`），其 1.57GB 惰性物化，已接受。
- 聚合吞吐到 ~2× 后受 GPU 固有算力（attention/gather）限制，非流式方案可优化范围。
- 诊断探针 `MLX_SERVE_EXPERT_STATS=N`（默认关、零开销）：每 N 次流式层访问打印 `hit / per_fill / disk / eval / copy_per_fill`。`copy_per_fill>0`＝捐赠退化成整池拷贝（golden 抓不到、这个能抓）。

## 文件清单

| 文件 | 状态 |
|---|---|
| `src/expert_store.zig` | 重写（新设计），git 有变更 |
| `src/{model,transformer,main,scheduler,server,mlx,tests}.zig` | 接线，git 有变更 |
| `tests/test_expert_streaming.sh` | 新建（运行时验收 harness） |
| `PROGRESS.md` | 本文件（已替换为准确状态） |
| `expert-streaming-handoff.md` | 只读依据，未改 |
