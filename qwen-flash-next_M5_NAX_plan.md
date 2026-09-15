# Qwen3.8-Flash-Next (qwen4_exp) on M5: NAX for the expert path

Hand-off plan for an M5 Max box. Everything below is UNMEASURED on M5. The M4 Max
numbers are the baseline you compare against, not a target.

## Where we are (M4 Max, 4-bit pack, 2026-08-27)

- Serial decode 62-63 tok/s, 8.5k context 55. MTP (`--mtp`): code 73-88, prose ~61, 8.5k ~55-58.
  MTP is opt-in because prose/long context are a wash or a loss. Auto depth ≈ fixed depth 2 on
  prose and 8.5k (same boot, 3 reps); the controller already plans m=2 there (`m_avg=2.00`).
- A verify row is BYTES: forward S=1 16.0 ms, S=2 22.4, S=4 30.6 (`MLX_SERVE_DECODE_FWD_UBENCH_S`),
  after the row-batched hc/GDN kernels (below). A depth-2 round is 2.05 serial forwards, so break-even
  needs >1.05 accepted per round; prose sits at 1.0. The second row's own experts are the cost.
- Batched decode (`--max-concurrent N`), with the row-batched kernels: prose 2 streams 78.4 aggregate
  (1.28x, was 1.14x), 4 streams 111 (1.81x, was 1.69x); 8.5k 2 streams 30.6/stream (was 28.4).
  qwen3_5 reaches 2.8x at 4; the gap is the MoE at N rows (sort path + N tokens' experts).
- Where a verify row's time goes on this arch: the routed expert banks. Dense pieces (attention q/k/v/o,
  hyper-connection read/write, QSA indexer, PLE key/value, shared expert, MTP head fc) already ride the
  `verifyQmm` lanes (split-K, wide tile, NAX m16 on G17). The experts go through `gather_qmm` /
  `gatherQmv` and have NO verify-width or NAX lane.
- `MtpCostProfile` for qwen4 has its own calibrated G17 surface since 2026-08-27/28 (item 1,
  landed — see below). It resolves only for the exact geometry it was measured on: affine 4-bit
  at **gs 64** with both head fc packs at that width. Any other pack — including a 4-bit **gs 32**
  qwen4 trunk — falls back to `.generic`, whose cost surface is a dense 27B GDN refit, i.e. it
  under-predicts a routed-expert verify row by 3-5x. Check which one is live before reading any
  MTP number off a box: `[mtp] cost profile g17-nax-qwen4-q4-gs64 engaged` in the boot log.

## What the 27B NAX lane taught us (do not skip)

- The lane is probe-gated (`naxAvailableFrom`: G17 + macOS >= 26.2 + `*_nax` in the metallib,
  `tests/test_mlx_staged_nax.sh` asserts the build). On M1-M4 it is inert; nothing can be validated there.
- First M5 report was -26% e2e. Correctness was clean; the loss was two CONTROLLER bugs (calibration
  cap, EV-seed default flip). Any regression report here needs the same two look-downs before
  touching the kernel.
- A verify lane is never byte-identical to stock: bars are fp32-dequant parity per width
  (`VerifyQmmParity`: gross ceiling 0.5, RMS ratio vs stock 3.0x, cosine), per-silicon slack keyed on
  `ane.chipBrandString`, plus `test_mtp_equivalence.sh` (greedy MTP == serial, near-ties acquitted at
  <= 0.15 nats).
- Every lever ships kill-switched, default OFF until a same-boot A/B on the target box says otherwise,
  with a one-shot `[...] engaged` log line so an arm can prove itself from its own log.

## Work items, in order

### 0. Baseline the box (half a day, no code)
1. Build: `./scripts/fetch-zig.sh`, `./scripts/build-mlx.sh`, `zig build -Doptimize=ReleaseFast`,
   `tests/test_mlx_staged_nax.sh` green, `./zig-out/bin/mlx-serve --version` shows NAX on.
2. Pack: `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` (~68 GB resident; the mixed-4-8bit pack needs ~75).
3. Same-boot serial vs MTP, 3 reps each, code/prose/8.5k prompts: `tests/qwen4_ab.sh mtp <tag>`
   (prompt fixtures in `tests/fixtures/qwen4_ab/`). Record `[spec-stats] mode=mtp` acceptance
   (`avg_per_round`) and `[vqmm] NAX verify lane engaged` lines. This tells you what the dense NAX
   lane already buys on this arch. `MLX_SERVE_DECODE_FWD_UBENCH=30 MLX_SERVE_DECODE_FWD_UBENCH_S=1|2|4`
   at load gives the per-row forward cost (M4: 16.0 / 22.4 / 30.6 ms).
4. Batched: `tests/qwen4_ab.sh batched <tag>` (1/2/4 streams of the prose prompt, aggregate tok/s).
5. `tests/test_qwen4_exp.sh` 38/38, `BATCHED_TEST_MODEL=<pack> tests/test_batched_equivalence.sh` 4/4,
   `MTP_FORCE_ENABLE=1 MTP_TEST_MODEL=<pack> tests/test_mtp_equivalence.sh` 11/11.
   If either is red on M5 before any change, stop and report; the NAX dense lane may already be
   engaging on shapes it never saw.
6. llmprobe, the number the PR is judged on: boot the pack (`--mtp`, `--prefix-cache-entries 0`) and
   run `npx llmprobe --bench-only` against it (`tests/bench.sh --only Flash-Next` does the boot + probe
   for you and prints the markdown row). Medians of 3. Current M5 Max, 4-bit pack, before this work —
   these are the numbers to beat, and the same run goes in the PR beside them:

   | cell | M5 Max today |
   |---|---|
   | decode | 91.2 tok/s (90.9 / 92.7 / 91.2) |
   | prefill | 1633 tok/s (1644.9 / 1633.1 / 1604.9) |
   | spec:predictable | 105.3 tok/s (105.7 / 105.3 / 104.2) |
   | spec:novel | 66.0 tok/s (60.9 / 66.4 / 65.9) |

   Same box, same pack, same flags, one boot per arm, or the comparison is worthless. A cell that
   moved by less than the spread of its three samples is noise, not a win.

### 1. Calibrated MTP cost profile for qwen4 on G17 — LANDED (do not redo)
Landed 2026-08-27/28, on this branch. What exists:
- `MtpCostProfile.g17_nax_qwen4_q4_gs64` — `src/mtp.zig:117`; the pure classifier
  `qwen4G17CostProfileForFingerprint` (`:169-180`, affine-4 + gs-64 + both head fc packs + a live
  NAX probe, else `.generic`); the live resolver `qwen4G17CostProfile` (`:206-226`); dispatch
  `MtpHeadRef.costProfile` (`src/generate.zig:381-386`).
- The EV surface `MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS` (`src/generate.zig:7524`) and its refit
  ledger (`:7498-7523`): M5 Max 40-core, 8.5k context, forced-depth saturated echo, depths
  {1,2,3,4,6} x two reversed passes x 3 reps — T(1)=25.41, T(2)=31.18, T(3)=35.70, T(4)=41.33,
  T(6)=53.58 ms over a 20.34 ms floor, composite marginals .257/.303. Wired at `:8293-8305`.
- The depth cap: `mtpDepthCapResolved` (`:7689-7708`) keeps this profile on
  `MTP_ADAPTIVE_DEFAULT_CAP` with its own written reason — depths 7-8 price at .345/pos against
  sub-60% tail acceptance, so there is no NAX region to reach. `adaptiveDepthCapForMachine` is NOT
  consulted for this profile (it only serves `.generic`).
- Tests: `src/mtp.zig:3641-3646`, `src/generate.zig:13775-13777`, `:13966-14021`.

So the plan's premise ("resolves to `.generic` on every machine") is stale. What is left here is
MEASUREMENT on the box, not code:
- Which profile is actually live for the pack under test. A 4-bit **gs 32** trunk resolves to
  `.generic` by design and inherits the dense-27B surface; that is a deliberate safety fallback
  ("off-profile geometry remains correct under generic", `src/mtp.zig:495`), not a bug — but it
  means a gs-32 pack needs its OWN measured row before its MTP numbers mean anything.
- Re-run S=1/2/4 forwards (`MLX_SERVE_DECODE_FWD_UBENCH*`) and compare against the marginals above.
  The controller is not the lever (the M4 twin in `docs/gotchas/engine-mlx.md`): a depth-2 round is
  2.05 serial forwards there. If the G17 ratio differs, refit the row; if it does not, leave it.
- Only after that, re-ask default-on (`Transformer.nativeMoeMtpHeadMeasured`,
  `src/transformer.zig:17729` — still a `return false` stub).

### 2. NAX-tiled grouped expert matmul for verify widths (the real lever)
Target shape: after routing, a verify of S rows x top-K gives S*K (row, expert) pairs; sort by expert
(`moeMLP2` already does the global sort for `B*S > 1`), so each expert sees a contiguous group of
1..S rows. Today that is `gather_qmm` sorted (`gatherExpertMm`) at S >= 2, `gatherQmv` at S == 1.

**Step 0 — the cheap arm, BEFORE any new kernel (landed, opt-in, unmeasured).** The fused rows
kernels (`gatherQmvGateUpRows` / `gatherQmvDownReduceRows`) are already a validated lane: no
`gather_qmm`, in-place bank reads, bit-identical per row to the solo single-row call, hermetic tests,
and the batched-decode shape already ships them. The verify shape is the SAME grouped (token,
expert) layout — one token's top-K per row — so the only thing keeping verify widths off it was the
`S == 1` guard in the one dispatch point, `moeDecodeDispatchArm` (`src/transformer.zig:35852`), which
an existing test pinned. `MLX_SERVE_MOE_VERIFY_ROWS=1` now lets B == 1, 2 <= S <= 8 reach it;
default OFF, `[moe] verify-width rows arm engaged: S=.. rows=.. topk=..` names the arm.

Run this FIRST, same boot, 3 reps, vs the sorted chain: MTP code/prose/8.5k with the lane on/off.
- If it wins => the sorted chain's `gather_qmm` addressing penalty was the cost, and the NAX tile
  below is a second-order refinement. Re-run the 2/4-stream aggregate too (same grouped shape).
- If it is flat or negative => record the number in `docs/gotchas/engine-mlx.md` and treat the NAX
  tile as unpromising: its premise was that the routed bank read is what a verify row pays for.
  Note the prior: the M4 side measured a per-ROW `gatherQmv` loop flat at S=2 / -11% at S=4
  (item 3), and three MoE fusions measured null on M4. The rows kernel is a different pattern at a
  different row origin, so it is worth one boot — but no more than one before demanding a signal.

- Then, only if step 0 moved things: new kernel beside `gatherQmvGateUpSource` / `gatherQmvDownReduceSource`:
  per expert group, an m16 NAX
  tile over the group's rows (pad to the tile; groups of 1-3 rows waste most of the tile, so the win
  is NOT guaranteed; measure). Gate+up fused like `gatherQmvGateUp`, down+reduce like
  `gatherQmvDownReduce`. Codegen with NAMED scalars (the M=8 plain-SIMD cliff and the "per-token
  template value = fresh JIT per value" trap both apply); `ShapeKey` cache keyed on FULL shape.
  A concrete design note for whoever builds it: the group's activation rows are CONTIGUOUS in the
  sorted pair order, so the NAX A-operand is the existing `x_rep` at row offset `g_start` (no gather
  needed); the run descriptor (start / length / expert) is derivable in-kernel from `sorted_inds`
  (a run starts where the expert changes), which avoids a data-dependent grid and any host sync —
  grid.x = total_inds, non-start threadgroups exit. Rows past the run must be masked on the STORE
  (`r < g_len`), and the activation buffer needs `BM-1 = 15` rows of tail padding so the last group's
  read cannot fault.
- Eligibility = the kernel's own conditions (4-bit affine gs64 first; q8 needs its own unpack arm,
  never let a templated `else` inherit q6), `K % 256 == 0`, `N % 32 == 0`, NAX probe live.
  Kill switch `MLX_SERVE_MOE_VERIFY_NAX=0`, default OFF until measured. One-shot
  `[moe-nax] verify lane engaged: S=.. groups=..` log.
- Parity: fp32 dequant ground truth per width S = 2..8, never kernel-vs-kernel, on the live dtype
  (bf16). Reuse `VerifyQmmParity` bars. Hermetic test beside `verifyQmm: split-K + msg + NAX ...`.
- A/B, same boot, 3 reps, interleaved arms by env on two ports if needed: MTP code/prose/8.5k with the
  lane on vs off. Report per-prompt and name the engine. If < +3% on code, it is null: record the
  number in `docs/gotchas/engine-mlx.md` and leave it OFF (three MoE fusions measured null on M4
  already; the GPU overlaps more than a dispatch count suggests).
- If it wins for verify, the same kernel serves batched decode (N slots x 1 row = the same grouped
  shape); re-run the 2/4-stream aggregate.

### 3. Batched-decode fused kernels at batch > 1 — OWNED BY THE M4 SIDE, landed 2026-08-27
Landed: hc read/write kernels walk `batch*seq` rows (`HC_FUSED_MAX_ROWS` 16), `gdnPreworkFused` +
`gdnNormGateFused` take a batch axis (`GDN_FUSED_MAX_ROWS` 16); all bit-identical per row to the
N=1 kernel (hermetic tests), `test_batched_equivalence.sh` 4/4. M4: S=2 forward 24.4 → 22.4 ms,
2/4-stream aggregate 1.14x → 1.28x / 1.69x → 1.81x. The MoE at N rows stays on the sort path: a
per-row `gatherQmv` loop was flat at S=2 and −11% at S=4, the multi-row gather kernel +38% before.
M5 only re-measures (`tests/qwen4_ab.sh batched`); the remaining lever is item 2's grouped kernel.

### 4. Only after 1-3 land: MTP default-on decision
`server.defaultEnableMtp` keeps MoE off; `Transformer.nativeMoeMtpHeadMeasured` is the exemption hook
and its bar is written in its doc comment: no losing cell on the context ladder vs serial, on prompts
that are not trivially draftable. Prose and 8.5k are the cells that have to stop losing.

## Traps specific to this arch (from the port)
- Every fixture is on a TINY random model (`tests/dump_qwen4_exp_fixtures.py`); it ties everywhere,
  `Qwen4Ties` acquits rows by the reference's own margins. Do not tighten bars on it.
- The spec "hidden" is the pre-mixer `[B,L,hc*hidden]` stream, not the mixed 2560.
- QSA keys/pooled blocks and the PLE window live on `SSMCacheEntry.aux_state`/`qsa_pooled`/`ple_prev`;
  the MTP head is NOT KV-only (`MtpCacheRef.kv()` null) so the prefix-cache spec-snap machinery skips it.
- An f32 scalar promotes the whole residual stream (`scalarOf`); `[dtype-trace] residual widened` is
  the tell. A diagnostics env read with `getenv != null` is armed by `=0` (`diagEnvOn`).
- Acceptance is a prompt-type property (code 0.91/0.80/0.68, prose 0.65/0.37/0.25 per index): measure
  it per index before pricing a round.
- Big models need ~45 s between boots on the same box or the second load fails preflight
  (`InsufficientMemory`), which looks like a code bug and is not. A harness that `pkill`s and boots
  on the same port must WAIT for the listener to go away (`qwen4_ab.sh` does) — "Port in use" lost a
  whole arm once.
- `--no-mtp` gates the in-checkpoint head only since 2026-08-27 (`entry.mtp` reads `params.mtp_enabled`);
  a control binary older than that runs MTP rounds on its `--no-mtp` baseline.
- `--no-vision` refuses image turns by name (`mediaRejectReason`); older builds answered them with a
  200 and no image. `test_qwen4_exp.sh` [11] pins it.
- The QSA prefill mask (`[S, kv]` per layer, 410 MB at 4096 x 25k) is billed by `server.qsaMaskBytes`;
  a 25k+ prompt on a tight box 400s at admission instead of a Metal OOM.
- MTP runs on image turns: the head takes the slot's M-RoPE table (`qwen4MtpForward(..., mrope_ctx)`).
  The vision fixture has no MTP reference; `test_qwen4_exp.sh` [7b] (tie-aware == serial) is the bar.

## Deliverables back
- The llmprobe `--bench-only` table from step 0.6, before AND after, in the PR description, plus
  `[spec-stats]` acceptance and the `[...] engaged` lines from the after arm's log.
- Numbers, same boot, 3 reps, per prompt, engine named, engaged lines quoted from the arm's own log.
- Kill switches for every new lane; defaults flipped only by a measured win on M5, and stated as
  M5-only (M1-M4 stay on the probe-gated path).
- Tests: hermetic parity per width + the live scripts above green; one rule line in `CLAUDE.md` and
  the story in `docs/gotchas/engine-mlx.md` per finding.
