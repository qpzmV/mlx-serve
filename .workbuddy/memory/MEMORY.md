# mlx-serve 项目长期笔记

## 构建与测试（易踩的坑）

- **工具链**：`./.zig-toolchain/zig`（0.17 nightly）。不要用系统的 zig。
- **`zig build test` 的测试根是 `src/tests.zig`，不编译 `main.zig`**。所以 `main.zig` 的语法/类型错误只有 `zig build -Doptimize=ReleaseFast` 才会暴露。改了 main.zig 一定要跑完整构建，别只跑单测。
- **单文件 `zig test src/xxx.zig` 会失败**：`error: no module named 'build_options'`。必须走 `zig build test`。想只跑一部分用 `zig build test -Dtest-filter="关键字"`（filter 只作用于主测试模块，vz_agent 那个模块不受影响）。
- **Zig 0.17 局部语法**：局部变量不能写 `///` 文档注释（只能 `//`）；`std.time.milliTimestamp` 不存在，用 `std.Io.Timestamp.now(io, .awake)`；dupe 用 `allocator.dupeSentinel(u8, s, 0)`。
- **`defer` 绑定包含它的块（block），不是函数**。把清理全局的 `defer` 写在赋值的 `{ }` 块内会立刻生效，且完全静默。见 `docs/gotchas/memory-pressure.md`。

## 内存压力 guard（2026-08-30 完成，commit 25ae8a6）

- 接线：`server.serve()` 内 `Scheduler.init` **之前** spawn 看门狗线程（加载期是内存最危险的时刻）；`src/mem_pressure.zig` 是纯状态机 `tick(now_ms, avail) → .none/.evict/.exit`。
- 默认水位自适应 `max(min(10GB, RAM/4), RAM/8)`；滞回 `max(min(2GB, RAM/32), RAM/64)`。128GB 机仍是 16GB。
- **测试必须用探针假数据**（`MLX_SERVE_MEM_PROBE_FILE`，文件内容为可用 RAM 的 GB 浮点）。用户明确反对制造真实内存压力（伤内存寿命）。
- 集成测试：`tests/test_mem_pressure.sh <model_dir> [port]`。可用小模型：
  `~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16/snapshots/6415d95f88be018ff9e46813119dc3bc12261328`
  （注意：`models--meta-llama--Llama-3.2-1B` 只有 tokenizer，没有权重，跑不了。）
- **写 shell 测试时注意 `check()` 的 1=通过 vs `grep -q` 的 0=找到**，直接传 `$?` 会把所有断言反向。脚本里已有 `grepok` 辅助函数。

## app 打包

- `FAST_DEV=1 bash app/build.sh` 在 Agent 沙箱里**跑不了**：Swift 工具链自带的 `sandbox-exec` 报 `sandbox_apply: Operation not permitted`（即便 `dangerouslyDisableSandbox=true`）。Swift 部分必须用户在自己终端编译。
- 只换二进制（Swift 层没改时够用），`app/MLX Core.app/` 是 gitignore 的：
  ```bash
  cp zig-out/bin/mlx-serve "app/MLX Core.app/Contents/MacOS/mlx-serve"
  install_name_tool -change "@rpath/libllama.dylib" "@executable_path/../Frameworks/libllama.dylib" "app/MLX Core.app/Contents/MacOS/mlx-serve"
  install_name_tool -change "@rpath/libmlxc.dylib"  "@executable_path/../Frameworks/libmlxc.dylib"  "app/MLX Core.app/Contents/MacOS/mlx-serve"
  install_name_tool -change "/opt/homebrew/opt/webp/lib/libwebp.7.dylib" "@executable_path/../Frameworks/libwebp.dylib" "app/MLX Core.app/Contents/MacOS/mlx-serve"
  codesign --force --sign - "app/MLX Core.app/Contents/MacOS/mlx-serve"
  ```
  **三个 dylib 都要改**，只改 libmlxc 会 `dyld: Library not loaded: @rpath/libllama.dylib`（exit 134）。

## qwen4_exp 专家流式（slotstream 移植，2026-08-31 交接文档已就绪，代码未写）

- 交接文档：`docs/expert-streaming-handoff.md`（8 节，含全部实测几何/代码落点/踩坑）。
- 目标：48 MoE 层 × 512 路由专家 = **77.07GB** 目前全物化装不下 128GB 机；移植 slotstream 的磁盘专家流式（槽池 + QD pread + gather_qmm 按槽位索引）。
- 实测几何（Vontra 检查点 `~/.mlx-serve/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit/`）：h=2560、ff=640、E=512、topK=10、48 层全 MoE、4-bit affine gs64；每专家 9 块共 **3.072MB**；非专家 36.14GB；ngram 表 29.8GB（页缓存，已实现于 `src/qwen4_exp.zig`，无需移植）；权重前缀 `language_model.model`→归一 `model`。
- **MLX 内核一行不改**：三条 MoE 路径（gather_qmm/gatherQmv/batched take）全经 `inds` 索引库，把「专家 ID」换「槽位」即可。唯一新代码 = `src/expert_store.zig`（ExpertStore+SlotPool+CheckpointIndex）+ `moeMLP2` 专家→槽位重映射 + 加载跳过 `.switch_mlp.` + CLI 旋钮。
- 关键落点：`moeMLP2` 17115 行（重映射插入点）、`initMoeLayers` 18914 行（switch_* 绑池）、`Transformer.init` 6886 行（构造池）、`shouldKeepWeightKey` 3466 行（加载跳过）、CLI 选项在 `main.zig` 669 行起（非 cli.zig）。
- 用户指令：先给文档自己写代码，**不要主动写代码**。

## 已知的环境性测试失败

- `vz_agent.test.e2e: a child does not inherit the agent's descriptors` 在 Agent 沙箱里必失败，与代码无关：沙箱注入 `TOYBOX_SANDBOX_SOCK` / `CODEBUDDY_SANDBOX_BROKER_IPC_ADDRESS`，exec 后的进程会多开一个 fd。空 env 下同样的 fork/close/exec 得到 `0 1 2 3 4`，继承 env 则 `0 1 2 3 4 5`。用户在自己终端跑应通过。
