//! Expert streaming for qwen4_exp — a disk-backed MoE slot pool.
//!
//! mlx-serve port of slotstream's `ExpertStore` + `SlotPool`. The qwen4_exp
//! trunk has `num_layers` MoE layers × `num_experts` routed experts, each a
//! 9-block quantized record (gate/up/down × weight/scales/biases) stacked
//! contiguously along axis 0 inside the safetensors shards. Materializing all of
//! them (≈77 GB) blows a 128 GB machine's RAM. Streaming keeps the experts ON
//! DISK and loads only the handful a token routes to into a fixed-size global
//! shared pool, then hands the pool's quantized banks to the SAME gather_qmm /
//! gatherQmv / batched-take kernels that run the non-streaming path — the only
//! change is that `rhs_indices` carry POOL SLOTS instead of EXPERT IDS.
//!
//! Load-bearing invariants (see expert-streaming-handoff.md):
//!   * ONE pool shared across all layers (CLOCK lets hot layers borrow slots from
//!     cold ones). NOT per-layer.
//!   * Each pool array is held by exactly ONE handle so MLX buffer donation fires
//!     (an extra reference silently turns every fill into a whole-pool copy).
//!     Hence `MoeMlpWeights.switch_*` stay null in streaming mode and are never
//!     bound to the pool banks; `moeMLP2` reads `pool.pools[p]` directly.
//!   * Slot ids fed to a SORTED gather path must be monotonic, so the
//!     expert→slot remap happens BEFORE the argsort (violating this yields
//!     silently wrong output on the gather_qmm fast path — no crash).
//!   * The pool map is keyed (layer, expert); one `moeMLP2` call is one layer.
//!   * Only the single inference thread touches the pool (the sole mlx caller), so
//!     map/CLOCK state is lock-free. A second mlx-calling thread must add a lock.
//!   * MTP draft experts (a 49th switch_mlp set) are NOT streamed — refs and the
//!     skip pattern only match `model.layers.{d}.mlp.switch_mlp.`, never `mtp.`.
//!
//! The `readPiece` gather is serial in v1 (a single `pread` lane hits ~9.5 GB/s
//! vs ~17.3 GB/s at QD8 — only 1.8× off, correct with zero thread complexity).
//! `MLX_SERVE_EXPERT_IO_QD` is reserved as the parallelization seam.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const is_darwin = builtin.os.tag == .macos or builtin.os.tag == .ios;

/// macOS-only fcntl commands (not exported by std.c). Best-effort: a failure to
/// set them is a page-cache-pollution concern, not a correctness one.
const F_NOCACHE: c_int = 55;
const F_RDAHEAD: c_int = 61;

pub const PIECES = 9;

/// The nine stacked banks per layer, in the fixed order the pool banks and the
/// per-expert record share: gate(w,s,b), up(w,s,b), down(w,s,b).
pub const PIECE_NAMES = [PIECES][]const u8{
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
};

/// Watermark/hysteresis mirrored from the memory-pressure guard by main.zig
/// (transformer.zig can't reach server.zig's global without an import cycle).
/// The pool budget must respect the USER'S configured watermark — a
/// `--memory-pressure 64` that leaves ~6 GiB of headroom must shrink the pool
/// accordingly, not a hardcoded default. 0 = use the machine defaults
/// (`PressureWatch.defaultWatermarkBytes/defaultHysteresisBytes`).
pub var g_watermark_bytes: u64 = 0;
pub var g_hysteresis_bytes: u64 = 0;

/// Minimum experts kept resident per layer (handoff §5.2 LOWER bound). Below this
/// the CLOCK thrashes and decode throughput collapses; the load-time budget clamp
/// raises to it, and the memory-pressure relief valve never shrinks past it.
pub const POOL_FLOOR_EXPERTS_PER_LAYER: u32 = 24;

/// The (layer, expert) pair a pool slot caches.
pub const ExpertKey = struct {
    layer: u16,
    expert: u32,

    pub fn bits(self: ExpertKey) u64 {
        return (@as(u64, self.layer) << 32) | @as(u64, self.expert);
    }
    pub fn fromBits(v: u64) ExpertKey {
        return .{ .layer = @intCast(v >> 32), .expert = @truncate(v) };
    }
};

/// Bytes per element for the dtypes a quantized expert bank can hold.
pub fn dtypeItemSize(d: mlx.mlx_dtype) u32 {
    return switch (d) {
        .uint32, .int32, .float32 => 4,
        .bfloat16, .float16 => 2,
        .uint8, .int8 => 1,
        else => @panic("expert_store: unexpected expert dtype"),
    };
}

/// One (layer, piece) stacked bank located inside a shard. Experts are
/// contiguous along axis 0, so expert `e` starts at `byte_offset + e*row_bytes`.
pub const TensorRef = struct {
    shard: u32, // index into ExpertStore.shards
    dtype: mlx.mlx_dtype,
    d1: i64, // shape[1]
    d2: i64, // shape[2]
    byte_offset: u64,
    row_bytes: u64,

    pub fn poolShape(self: TensorRef, n: c_int) [3]c_int {
        return .{ n, @intCast(self.d1), @intCast(self.d2) };
    }
};

pub const Shard = struct {
    path: [:0]u8, // allocator-owned
    fd: std.c.fd_t,
};

/// True when `key` is a trunk routed-expert tensor
/// (`[<anything>.]model.layers.{d}.mlp.switch_mlp.…`). `mtp.` keys are rejected
/// (they stay resident). Assumes qwen4 + streaming already established.
pub fn isTrunkSwitchKey(key: []const u8) bool {
    if (std.mem.indexOf(u8, key, ".mtp.") != null) return false;
    if (std.mem.indexOf(u8, key, ".mlp.switch_mlp.") == null) return false;
    const li = std.mem.indexOf(u8, key, "model.layers.") orelse return false;
    if (li != 0 and key[li - 1] != '.') return false;
    return true;
}

pub const ExpertStoreError = error{
    MissingShardIndex,
    MissingExpertTensor,
    GeometryMismatch,
    IoReadFailed,
};

/// Parses the safetensors headers of the shards holding the trunk switch_mlp
/// banks and resolves every (layer, piece) to a `TensorRef`.
pub const ExpertStore = struct {
    allocator: std.mem.Allocator,
    num_layers: u32,
    num_experts: u32,
    /// Sum of the 9 per-expert row_bytes = one expert's full record size.
    record_bytes: u64,
    piece_row_bytes: [PIECES]u64 = undefined,
    shards: []Shard = &.{},
    /// [num_layers * PIECES] refs, indexed `layer*PIECES + piece`.
    refs: []TensorRef = &.{},

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        num_layers: u32,
        num_experts: u32,
    ) !*ExpertStore {
        const self = try allocator.create(ExpertStore);
        self.* = .{
            .allocator = allocator,
            .num_layers = num_layers,
            .num_experts = num_experts,
            .record_bytes = 0,
        };
        errdefer self.deinit();

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();

        // 1. The index names every key's shard file; collect the distinct shards
        //    that hold at least one trunk switch_mlp tensor.
        const idx_path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors.index.json", .{model_dir}, 0);
        const idx_bytes = readFileAlloc(io, a, idx_path) catch |err| {
            log.err("expert_store: cannot read {s}: {s}\n", .{ idx_path, @errorName(err) });
            return ExpertStoreError.MissingShardIndex;
        };
        const idx_json = try std.json.parseFromSliceLeaky(std.json.Value, a, idx_bytes, .{});
        const weight_map_obj = idx_json.object.get("weight_map") orelse return ExpertStoreError.MissingShardIndex;

        var shard_names = std.StringHashMap(u32).init(a);
        var shard_list: std.ArrayList([]const u8) = .empty;
        var it = weight_map_obj.object.iterator();
        while (it.next()) |e| {
            if (!isTrunkSwitchKey(e.key_ptr.*)) continue;
            const shard_name = e.value_ptr.string;
            const gop = try shard_names.getOrPut(shard_name);
            if (!gop.found_existing) {
                gop.value_ptr.* = @intCast(shard_list.items.len);
                try shard_list.append(a, shard_name);
            }
        }
        if (shard_list.items.len == 0) return ExpertStoreError.MissingShardIndex;

        // 2. Open each involved shard (F_NOCACHE so cold expert bytes don't
        //    evict the ngram table's page cache).
        self.shards = try allocator.alloc(Shard, shard_list.items.len);
        for (self.shards) |*sh| sh.* = .{ .path = undefined, .fd = -1 };
        errdefer for (self.shards) |sh| {
            if (sh.fd >= 0) {
                _ = std.c.close(sh.fd);
                allocator.free(sh.path);
            }
        };
        for (self.shards, 0..) |*sh, i| {
            sh.path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, shard_list.items[i] }, 0);
            sh.fd = std.c.open(sh.path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
            if (sh.fd < 0) {
                log.err("expert_store: open failed {s}\n", .{sh.path});
                allocator.free(sh.path);
                sh.path = undefined;
                return ExpertStoreError.MissingShardIndex;
            }
            if (comptime is_darwin) {
                _ = std.c.fcntl(sh.fd, F_NOCACHE, @as(c_int, 1));
                _ = std.c.fcntl(sh.fd, F_RDAHEAD, @as(c_int, 0));
            }
        }

        // 3. Parse each shard header once, record refs for its trunk tensors.
        self.refs = try allocator.alloc(TensorRef, num_layers * PIECES);
        @memset(self.refs, std.mem.zeroes(TensorRef));
        var have = try allocator.alloc(bool, num_layers * PIECES);
        defer allocator.free(have);
        @memset(have, false);

        var hdr_buf: std.ArrayList(u8) = .empty;
        for (self.shards, 0..) |*sh, sh_i| {
            var len8: [8]u8 = undefined;
            if (!preadFull(sh.fd, &len8, 0)) return ExpertStoreError.MissingShardIndex;
            const hlen: u64 = std.mem.readInt(u64, len8[0..8], .little);
            try hdr_buf.resize(a, @intCast(hlen));
            if (!preadFull(sh.fd, hdr_buf.items, 8)) return ExpertStoreError.MissingShardIndex;
            const hdr = try std.json.parseFromSliceLeaky(std.json.Value, a, hdr_buf.items, .{});
            const data_off: u64 = 8 + hlen;
            var hit_it = hdr.object.iterator();
            while (hit_it.next()) |e| {
                if (!isTrunkSwitchKey(e.key_ptr.*)) continue;
                const lp = parseTrunkKey(e.key_ptr.*) orelse continue;
                if (lp.layer >= num_layers) continue; // defensive: out-of-range layer
                const obj = e.value_ptr.object;
                const dt = safetensorsDtype(obj.get("dtype").?.string) orelse continue;
                const shape = obj.get("shape").?.array.items;
                const offs = obj.get("data_offsets").?.array.items;
                const d1: i64 = if (shape.len >= 2) shape[1].integer else 1;
                const d2: i64 = if (shape.len >= 3) shape[2].integer else 1;
                const row_bytes = @as(u64, @intCast(d1 * d2 * @as(i64, @intCast(dtypeItemSize(dt)))));
                const idx = lp.layer * PIECES + lp.piece;
                self.refs[idx] = .{
                    .shard = @intCast(sh_i),
                    .dtype = dt,
                    .d1 = d1,
                    .d2 = d2,
                    .byte_offset = data_off + @as(u64, @intCast(offs[0].integer)),
                    .row_bytes = row_bytes,
                };
                have[idx] = true;
            }
        }

        // 4. Completeness + geometry check (uniform MoE: every layer shares layer
        //    0's per-piece geometry). Never hard-code a record size — it comes
        //    from the header (handoff §踩坑14).
        for (0..PIECES) |p| {
            if (!have[p]) return ExpertStoreError.MissingExpertTensor;
            self.piece_row_bytes[p] = self.refs[p].row_bytes;
        }
        for (0..num_layers) |l| {
            for (0..PIECES) |p| {
                if (!have[l * PIECES + p]) {
                    log.err("expert_store: missing layer {d} piece {s}\n", .{ l, PIECE_NAMES[p] });
                    return ExpertStoreError.MissingExpertTensor;
                }
                if (self.refs[l * PIECES + p].row_bytes != self.piece_row_bytes[p])
                    return ExpertStoreError.GeometryMismatch;
            }
        }
        var total: u64 = 0;
        for (self.piece_row_bytes) |rb| total += rb;
        self.record_bytes = total;
        log.info("[expert] store: {d} layers × {d} experts, record {d:.3} MB(dec), piece bytes [{d},{d},{d}]\n", .{
            num_layers, num_experts, @as(f64, @floatFromInt(total)) / 1_000_000.0,
            self.piece_row_bytes[0], self.piece_row_bytes[1], self.piece_row_bytes[6],
        });
        return self;
    }

    pub fn deinit(self: *ExpertStore) void {
        const a = self.allocator;
        for (self.shards) |*sh| {
            if (sh.fd >= 0) {
                _ = std.c.close(sh.fd);
                a.free(sh.path);
            }
        }
        if (self.shards.len > 0) a.free(self.shards);
        if (self.refs.len > 0) a.free(self.refs);
        a.destroy(self);
    }

    /// Read `experts`' records for one piece into `dst` (`experts.len * row_bytes`
    /// long), row-major in the order given. False on any short read.
    pub fn readPiece(self: *const ExpertStore, layer: u16, piece: usize, experts: []const u32, dst: []u8) bool {
        const r = self.refs[layer * PIECES + piece];
        std.debug.assert(dst.len == experts.len * r.row_bytes);
        var i: usize = 0;
        while (i < experts.len) : (i += 1) {
            std.debug.assert(experts[i] < self.num_experts);
            const off = r.byte_offset + @as(u64, experts[i]) * r.row_bytes;
            if (!preadFull(self.shards[r.shard].fd, dst[i * r.row_bytes ..][0..r.row_bytes], off)) return false;
        }
        return true;
    }
};

/// The slot pool: `slots` quantized expert banks + a CLOCK eviction map.
pub const SlotPool = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    store: *ExpertStore,
    slots: u32,

    pools: [PIECES]mlx.mlx_array = undefined,

    map: std.AutoHashMap(u64, u32), // ExpertKey.u64 -> slot
    key_of: []?u64 = &.{}, // slot -> ExpertKey.u64 (null = never filled)
    ref: []bool = &.{}, // second-chance clock bit
    pin: []bool = &.{}, // pinned for the current layer's forward
    hand: u32 = 0,

    hits: u64 = 0,
    misses: u64 = 0,
    /// Diagnostics for the relief/donation judgement (single inference thread, so
    /// plain u64 is enough — no atomics). `fill_ops` = expert-rows scattered in,
    /// `io_bytes` = bytes pread, `fill_ns` = cumulative ns inside ensure()'s read+
    /// scatter section. A broken buffer-donation shows as fill_ns per op jumping
    /// toward a whole-pool copy (~ms/row) instead of ~tens of µs.
    fill_ops: u64 = 0,
    io_bytes: u64 = 0,
    fill_ns: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        s: mlx.mlx_stream,
        store: *ExpertStore,
        slots: u32,
    ) !*SlotPool {
        std.debug.assert(slots > 0);
        const self = try allocator.create(SlotPool);
        self.* = .{
            .allocator = allocator,
            .s = s,
            .store = store,
            .slots = slots,
            .map = std.AutoHashMap(u64, u32).init(allocator),
        };
        errdefer self.deinit();

        for (0..PIECES) |p| {
            const shape = store.refs[p].poolShape(@intCast(slots));
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, &shape, shape.len, store.refs[p].dtype, s));
            self.pools[p] = arr; // exactly one handle from here on
        }
        self.key_of = try allocator.alloc(?u64, slots);
        errdefer allocator.free(self.key_of);
        @memset(self.key_of, null);
        self.ref = try allocator.alloc(bool, slots);
        errdefer allocator.free(self.ref);
        @memset(self.ref, false);
        self.pin = try allocator.alloc(bool, slots);
        errdefer allocator.free(self.pin);
        @memset(self.pin, false);
        return self;
    }

    pub fn deinit(self: *SlotPool) void {
        const a = self.allocator;
        for (&self.pools) |*arr| {
            if (arr.ctx != null) _ = mlx.mlx_array_free(arr.*);
        }
        self.map.deinit();
        if (self.key_of.len > 0) a.free(self.key_of);
        if (self.ref.len > 0) a.free(self.ref);
        if (self.pin.len > 0) a.free(self.pin);
        a.destroy(self);
    }

    /// CLOCK hand walk: skip pinned; clear a set ref bit and advance; take the
    /// first slot with a clear ref bit. `scanned < 3*slots` guards livelock (the
    /// pool must fit one layer's unique expert set — sized by budget clamp).
    fn victim(self: *SlotPool) u32 {
        var scanned: u32 = 0;
        while (scanned < 3 *| self.slots) : (scanned += 1) {
            const i = self.hand;
            self.hand = (self.hand + 1) % self.slots;
            if (self.pin[i]) continue;
            if (self.ref[i]) {
                self.ref[i] = false;
                continue;
            }
            return @intCast(i);
        }
        @panic("expert_store: CLOCK found no victim — raise --experts-per-layer / pool size");
    }

    /// Look up or load `experts` (unique, same layer), writing each input's slot
    /// id to `out_slots[i]`. Misses are pread from disk and scattered into the
    /// pool banks (buffer donation → only touched rows written). Caller must have
    /// deduped `experts`.
    pub fn ensure(self: *SlotPool, layer: u16, experts: []const u32, out_slots: []u32) !void {
        std.debug.assert(experts.len == out_slots.len);
        if (experts.len == 0) return;

        const a = self.allocator;
        var miss_pos: std.ArrayList(usize) = .empty;
        defer miss_pos.deinit(a);
        const miss_experts = try a.alloc(u32, experts.len);
        defer a.free(miss_experts);
        const chosen = try a.alloc(u32, experts.len);
        defer a.free(chosen);

        for (experts, 0..) |e, i| {
            const k = (ExpertKey{ .layer = layer, .expert = e }).bits();
            if (self.map.get(k)) |sl| {
                out_slots[i] = sl;
                self.ref[sl] = true;
                self.pin[sl] = true;
                self.hits += 1;
            } else {
                try miss_pos.append(a, i);
            }
        }
        const nm = miss_pos.items.len;
        if (nm == 0) return;

        for (miss_pos.items, 0..) |ei, m| {
            const sl = self.victim();
            if (self.key_of[sl]) |oldk| _ = self.map.remove(oldk);
            chosen[m] = sl;
            miss_experts[m] = experts[ei];
        }

        // Slot-index array (u32 [nm]) as one MLX index tensor for all scatters.
        // `chosen` already holds the victim slots in miss order — reuse it.
        const sidx_shape = [_]c_int{@intCast(nm)};
        const slot_idx_arr = mlx.mlx_array_new_data(chosen.ptr, &sidx_shape, 1, .uint32);
        defer _ = mlx.mlx_array_free(slot_idx_arr);
        const idx_vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{slot_idx_arr}, 1);
        defer _ = mlx.mlx_vector_array_free(idx_vec);

        const fill_io = std.Io.Threaded.global_single_threaded.io();
        const fill_t0 = std.Io.Timestamp.now(fill_io, .awake);
        for (0..PIECES) |p| {
            const rb = self.store.piece_row_bytes[p];
            const staging = try a.alloc(u8, nm * rb);
            defer a.free(staging);
            if (!self.store.readPiece(layer, p, miss_experts[0..nm], staging)) return error.IoReadFailed;

            const r = self.store.refs[layer * PIECES + p];
            // MLX general scatter wants updates.ndim == a.ndim + indices.ndim:
            // for a 1-D [nm] index that is [nm, 1, d1, d2] — the singleton axis
            // mlx_scatter_args_array inserts (indices.shape ++ [1] ++ src.shape[1:]).
            // The staging bytes are contiguous and row-major by miss order, so
            // this is a pure shape relabel, not a data move.
            const upd_shape = [_]c_int{ @intCast(nm), 1, @intCast(r.d1), @intCast(r.d2) };
            const updates = mlx.mlx_array_new_data(staging.ptr, &upd_shape, upd_shape.len, r.dtype);
            defer _ = mlx.mlx_array_free(updates);

            const axes = [_]c_int{0};
            var updated = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_scatter(&updated, self.pools[p], idx_vec, updates, &axes, 1, self.s));
            // Force the op NOW (staging/updates still alive), then donate-swap the
            // pool handle: old freed, `updated` is the sole reference for next fill.
            try mlx.check(mlx.mlx_array_eval(updated));
            _ = mlx.mlx_array_free(self.pools[p]);
            self.pools[p] = updated;
        }
        self.fill_ns += @intCast(fill_t0.untilNow(fill_io, .awake).nanoseconds);
        self.fill_ops += nm;
        self.io_bytes += nm * self.store.record_bytes;

        for (miss_pos.items, chosen[0..nm]) |ei, sl| {
            const k = (ExpertKey{ .layer = layer, .expert = experts[ei] }).bits();
            self.key_of[sl] = k;
            try self.map.put(k, sl);
            self.ref[sl] = true;
            self.pin[sl] = true;
            out_slots[ei] = sl;
            self.misses += 1;
        }
    }

    /// Release all pins after a layer's forward so the next layer can evict.
    pub fn unpinAll(self: *SlotPool) void {
        @memset(self.pin, false);
    }

    /// Rebuild at a new size (memory-pressure relief valve). Cold restart: old
    /// contents + the whole map are dropped — stale key→slot mappings would
    /// silently return wrong experts. Always clears the MLX cache so a shrink
    /// actually returns bytes to the OS (else the watchdog still exits).
    pub fn resize(self: *SlotPool, new_slots: u32) !void {
        std.debug.assert(new_slots > 0);
        if (new_slots == self.slots) return;
        const a = self.allocator;
        for (&self.pools) |*arr| {
            if (arr.ctx != null) _ = mlx.mlx_array_free(arr.*);
        }
        a.free(self.key_of);
        a.free(self.ref);
        a.free(self.pin);
        self.map.clearAndFree();

        self.slots = new_slots;
        for (0..PIECES) |p| {
            const shape = self.store.refs[p].poolShape(@intCast(new_slots));
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, &shape, shape.len, self.store.refs[p].dtype, self.s));
            self.pools[p] = arr;
        }
        self.key_of = try a.alloc(?u64, new_slots);
        @memset(self.key_of, null);
        self.ref = try a.alloc(bool, new_slots);
        @memset(self.ref, false);
        self.pin = try a.alloc(bool, new_slots);
        @memset(self.pin, false);
        self.hand = 0;
        _ = mlx.mlx_clear_cache();
    }

    pub fn hitRate(self: *const SlotPool) f64 {
        const tot = self.hits + self.misses;
        if (tot == 0) return 0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(tot));
    }
};

// ── helpers ──

fn preadFull(fd: std.c.fd_t, dst: []u8, offset: u64) bool {
    var done: usize = 0;
    while (done < dst.len) {
        const rc = std.c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
        if (rc <= 0) return false;
        done += @intCast(rc);
    }
    return true;
}

fn readFileAlloc(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [8192]u8 = undefined;
    var rs = file.reader(io, &buf);
    return rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
}

const LayerPiece = struct { layer: u32, piece: usize };

/// Parse "…model.layers.{d}.mlp.switch_mlp.{proj}.{field}" into (layer, pieceIdx).
fn parseTrunkKey(key: []const u8) ?LayerPiece {
    const needle = "model.layers.";
    const li = std.mem.indexOf(u8, key, needle) orelse return null;
    const after = key[li + needle.len ..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    const tail = after[dot + 1 ..];
    const sm = std.mem.indexOf(u8, tail, ".switch_mlp.") orelse return null;
    const piece_str = tail[sm + ".switch_mlp.".len ..];
    var p: usize = 0;
    while (p < PIECES) : (p += 1) {
        if (std.mem.eql(u8, piece_str, PIECE_NAMES[p])) return .{ .layer = layer, .piece = p };
    }
    return null;
}

fn safetensorsDtype(s: []const u8) ?mlx.mlx_dtype {
    if (std.mem.eql(u8, s, "U32")) return .uint32;
    if (std.mem.eql(u8, s, "BF16")) return .bfloat16;
    if (std.mem.eql(u8, s, "F16")) return .float16;
    if (std.mem.eql(u8, s, "F32")) return .float32;
    if (std.mem.eql(u8, s, "U8")) return .uint8;
    if (std.mem.eql(u8, s, "I32")) return .int32;
    return null;
}

// ── tests (pure logic; no GPU / no MLX hot path) ──

const testing = std.testing;

test "isTrunkSwitchKey keeps mtp, matches trunk layers only" {
    try testing.expect(isTrunkSwitchKey("language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight"));
    try testing.expect(isTrunkSwitchKey("model.layers.0.mlp.switch_mlp.down_proj.biases"));
    try testing.expect(!isTrunkSwitchKey("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"));
    try testing.expect(!isTrunkSwitchKey("model.layers.7.mlp.shared_expert.gate_proj.weight"));
    try testing.expect(!isTrunkSwitchKey("model.layers.7.mlp.gate.weight"));
    try testing.expect(!isTrunkSwitchKey("lm_head.weight"));
}

test "parseTrunkKey resolves layer and piece order" {
    const a = parseTrunkKey("language_model.model.layers.12.mlp.switch_mlp.up_proj.scales").?;
    try testing.expectEqual(@as(u32, 12), a.layer);
    try testing.expectEqual(@as(usize, 4), a.piece);
    const b = parseTrunkKey("model.layers.0.mlp.switch_mlp.down_proj.biases").?;
    try testing.expectEqual(@as(u32, 0), b.layer);
    try testing.expectEqual(@as(usize, 8), b.piece);
    try testing.expect(parseTrunkKey("model.layers.x.mlp.switch_mlp.gate_proj.weight") == null);
}

test "ExpertKey round-trips through u64" {
    const k = ExpertKey{ .layer = 47, .expert = 511 };
    try testing.expectEqual(k, ExpertKey.fromBits(k.bits()));
    try testing.expect(!std.meta.eql(ExpertKey.fromBits(k.bits()), ExpertKey{ .layer = 47, .expert = 510 }));
    // layer and expert must not alias in the u64 key:
    try testing.expect(k.bits() != (ExpertKey{ .layer = 48, .expert = 511 }).bits());
}

test "dtype item sizes" {
    try testing.expectEqual(@as(u32, 4), dtypeItemSize(.uint32));
    try testing.expectEqual(@as(u32, 2), dtypeItemSize(.bfloat16));
}
