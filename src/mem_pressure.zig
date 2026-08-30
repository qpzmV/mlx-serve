//! Machine-wide memory pressure guard (feature: `--memory-pressure`).
//!
//! A pure state machine over `(time, available RAM)` pairs: `tick` returns the
//! action the inference thread should take NOW, given the current monotonic
//! time and the current machine-wide available RAM. It reads no memory,
//! allocates nothing, and calls no MLX/Metal API — the unit tests drive it
//! with hand-written `(time, RAM)` pairs, so the whole guard is testable
//! without touching a byte of real RAM.
//!
//! Semantics (all three pinned by the tests at the bottom):
//!   * OFF (`watermark_bytes == 0`) → `.none` forever, zero cost.
//!   * first check under the bar → `.evict` (unload ONE resident model) and
//!     the exit countdown starts;
//!   * sustained pressure ≥ `exit_after_ms` → `.exit` (the process quits to
//!     give its RAM back — a Mac has no OOM killer, this guard is ours);
//!   * recovery to or above `watermark + hysteresis` → countdown RESETS (the
//!     band between the bar and the recovery bar keeps the countdown
//!     counting: hovering just under the bar IS sustained pressure);
//!   * an unknown reading (unreadable probe file) is the ABSENCE of
//!     information, not a reading — it neither starts nor resets anything.
//!
//! "Available" is `status.getAvailableMemBytes()`'s definition:
//! `total − (wired + compressed + anonymous-resident − purgeable)`, NOT free
//! pages (macOS keeps free ≈ 0 by design; a free-page bar would fire the
//! guard on a healthy machine).

const std = @import("std");
const log = @import("log.zig");
const status = @import("status.zig");

const GB: u64 = 1024 * 1024 * 1024;

pub const Action = enum { none, evict, exit };

pub const PressureWatch = struct {
    /// 0 = guard disabled. Default when enabled: `max(10 GB, RAM/8)` —
    /// 16 GB on a 128 GB machine, where the resident model + its prefill
    /// working set legitimately needs most of the RAM.
    watermark_bytes: u64 = 0,
    /// Recovery bar = watermark + this. Below the bar the countdown counts;
    /// only climbing back over the recovery bar proves the pressure passed.
    hysteresis_bytes: u64 = 0,
    /// Sustained-pressure milliseconds until `.exit`. 0 = evict-only, never
    /// exit (`--memory-pressure-exit-after 0`).
    exit_after_ms: i64 = 30_000,
    /// Check throttle: at most one real RAM read per interval, so a busy
    /// decode loop never pays for the guard between rounds.
    check_interval_ms: i64 = 5_000,
    /// Test/ops injection point (`MLX_SERVE_MEM_PROBE_FILE`): a file holding
    /// the available RAM in GB as a float (e.g. "9.0"). When present it
    /// REPLACES the real read — the integration tests drive the whole guard
    /// through it and never consume a byte of RAM.
    probe_file: ?[:0]const u8 = null,
    probe_unreadable_logged: bool = false,

    last_check: ?i64 = null,
    /// When the available-RAM reading first dipped under the watermark.
    /// `null` = not currently under (window closed). A nullable is deliberate:
    /// a `0`-sentinel collides with `now_ms == 0`, leaving the window
    /// permanently "open" at t0 and the exit countdown never able to start.
    below_since_ms: ?i64 = null,

    pub fn defaultWatermarkBytes(total: u64) u64 {
        if (total == 0) return 0;
        return @max(10 * GB, total / 8);
    }

    pub fn defaultHysteresisBytes(total: u64) u64 {
        if (total == 0) return 0;
        return @max(2 * GB, total / 64);
    }

    /// The current available-RAM reading: probe file (GB, float) when set,
    /// otherwise the real machine read. `maxInt` = unknown (unreadable probe
    /// file) — callers must treat it as "no information", never as pressure
    /// OR as recovery.
    pub fn availableBytes(self: *PressureWatch, io: std.Io, gpa: std.mem.Allocator) u64 {
        const path = self.probe_file orelse return status.getAvailableMemBytes();
        const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch {
            if (!self.probe_unreadable_logged) {
                self.probe_unreadable_logged = true;
                log.warn("[mem-pressure] probe file {s} unreadable; guard idles (naming this once)\n", .{path});
            }
            return std.math.maxInt(u64);
        };
        defer f.close(io);
        var rb: [64]u8 = undefined;
        var rs = f.reader(io, &rb);
        const content = rs.interface.allocRemaining(gpa, .limited(32)) catch {
            if (!self.probe_unreadable_logged) {
                self.probe_unreadable_logged = true;
                log.warn("[mem-pressure] probe file {s} unreadable; guard idles (naming this once)\n", .{path});
            }
            return std.math.maxInt(u64);
        };
        defer gpa.free(content);
        const gb = std.fmt.parseFloat(f64, std.mem.trim(u8, content, " \r\n\t")) catch return std.math.maxInt(u64);
        if (!(gb > 0) or gb > 4096) return std.math.maxInt(u64); // NaN, <=0, absurd → unknown
        return @intFromFloat(gb * 1_073_741_824.0); // GiB-scale input, byte output
    }

    /// Pure step: given the monotonic clock and the available RAM, decide the
    /// action. Time and memory are INPUTS — nothing here reads clocks or
    /// memory, so every transition is unit-testable with literals.
    pub fn tick(self: *PressureWatch, now_ms: i64, avail_bytes: u64) Action {
        if (self.watermark_bytes == 0) return .none;
        if (self.last_check) |last| {
            if (now_ms - last < self.check_interval_ms) return .none;
        }
        self.last_check = now_ms;
        if (avail_bytes == std.math.maxInt(u64)) return .none; // unknown ≠ pressure, ≠ recovery
        if (avail_bytes >= self.watermark_bytes +| self.hysteresis_bytes) {
            if (self.below_since_ms != null) {
                self.below_since_ms = null;
                log.info("[mem-pressure] available RAM back above the recovery bar; exit countdown reset\n", .{});
            }
            return .none;
        }
        if (avail_bytes < self.watermark_bytes) {
            if (self.below_since_ms) |since| {
                if (self.exit_after_ms > 0 and now_ms - since >= self.exit_after_ms) return .exit;
                return .none;
            }
            // First dip under the bar: open the window and evict one model.
            self.below_since_ms = now_ms;
            return .evict;
        }
        return .none; // between the bar and the recovery band: countdown keeps counting
    }
};

// ── unit tests ───────────────────────────────────────────────────────────
// Every case drives the machine with hand-written (time, RAM) pairs. No test
// here allocates, touches, or pins a single byte of RAM — that is the point
// of the injection design.

const t = std.testing;
const GBf: u64 = GB; // readability in the literals below

test "guard OFF is free: watermark 0 answers .none for any input, including a dead machine" {
    var w = PressureWatch{};
    try t.expectEqual(@as(Action, .none), w.tick(0, 0));
    try t.expectEqual(@as(Action, .none), w.tick(10 * @as(i64, 60_000), 0));
}

test "first dip under the bar evicts one; sustained pressure exits" {
    var w = PressureWatch{ .watermark_bytes = 10 * GBf, .hysteresis_bytes = 2 * GBf };
    try t.expectEqual(@as(Action, .evict), w.tick(0, 9 * GBf)); // countdown starts
    try t.expectEqual(@as(Action, .none), w.tick(10_000, 9 * GBf)); // counting down
    try t.expectEqual(@as(Action, .exit), w.tick(31_000, 9 * GBf)); // 31 s ≥ 30 s bar
}

test "recovery over the recovery bar resets the countdown; a fresh dip restarts it" {
    var w = PressureWatch{ .watermark_bytes = 10 * GBf, .hysteresis_bytes = 2 * GBf, .exit_after_ms = 10_000 };
    try t.expectEqual(@as(Action, .evict), w.tick(0, 9 * GBf));
    try t.expectEqual(@as(Action, .none), w.tick(5_000, 13 * GBf)); // recovered (≥ 12 GB band top)
    // 20 s after the FIRST dip a sticky countdown would already be `.exit`;
    // the reset must land as a fresh `.evict` instead:
    try t.expectEqual(@as(Action, .evict), w.tick(20_000, 9 * GBf));
}

test "in-band values keep the countdown counting (sticky band)" {
    var w = PressureWatch{ .watermark_bytes = 10 * GBf, .hysteresis_bytes = 2 * GBf, .exit_after_ms = 10_000 };
    try t.expectEqual(@as(Action, .evict), w.tick(0, 9 * GBf));
    try t.expectEqual(@as(Action, .none), w.tick(5_000, 11 * GBf)); // inside the band: not a recovery
    try t.expectEqual(@as(Action, .exit), w.tick(11_000, 9 * GBf)); // countdown ran through the band
}

test "check throttle: two wakes inside one interval are ONE check" {
    var w = PressureWatch{ .watermark_bytes = 10 * GBf, .exit_after_ms = 10_000 };
    try t.expectEqual(@as(Action, .evict), w.tick(0, 9 * GBf)); // t0: window opens, last_check = 0
    try t.expectEqual(@as(Action, .none), w.tick(5, 9 * GBf)); // 5 ms since the last check: no check
    try t.expectEqual(@as(Action, .exit), w.tick(10_001, 9 * GBf)); // 10 s later: a real check, countdown ran out
}

test "unknown availability (maxInt) is no information: it neither starts nor resets" {
    var w = PressureWatch{ .watermark_bytes = 10 * GBf, .hysteresis_bytes = 2 * GBf, .exit_after_ms = 10_000 };
    try t.expectEqual(@as(Action, .evict), w.tick(0, 9 * GBf));
    try t.expectEqual(@as(Action, .none), w.tick(5_000, std.math.maxInt(u64))); // unknown
    // a sticky countdown treats unknown as "no news" — the next below-bar
    // reading still counts from t0 and exits:
    try t.expectEqual(@as(Action, .exit), w.tick(11_000, 9 * GBf));
}

test "defaults are per-machine: max(10 GB, RAM/8) and max(2 GB, RAM/64)" {
    try t.expectEqual(@as(u64, 0), PressureWatch.defaultWatermarkBytes(0));
    try t.expectEqual(10 * GBf, PressureWatch.defaultWatermarkBytes(16 * GBf)); // 16/8=2 < 10 → floor
    try t.expectEqual(16 * GBf, PressureWatch.defaultWatermarkBytes(128 * GBf)); // 128/8=16 → scale
    try t.expectEqual(2 * GBf, PressureWatch.defaultHysteresisBytes(128 * GBf));
    try t.expectEqual(4 * GBf, PressureWatch.defaultHysteresisBytes(256 * GBf));
}

test "probe file: the injected number IS the reading; missing/garbage is unknown" {
    const io = std.testing.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "probe", .data = "9.0" });
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.SkipZigTest;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const probe_tmp = try std.fmt.allocPrint(t.allocator, "{s}/.zig-cache/tmp/{s}/probe", .{ cwd, tmp.sub_path });
    defer t.allocator.free(probe_tmp);
    const probe = try t.allocator.dupeSentinel(u8, probe_tmp, 0);
    defer t.allocator.free(probe);

    var w = PressureWatch{ .watermark_bytes = 10 * GBf, .probe_file = probe };
    const avail = w.availableBytes(io, t.allocator);
    // 9.0 GiB injected → read back as exactly 9 GiB of bytes:
    try t.expectEqual(@as(u64, 9 * 1_073_741_824), avail);
    // …and it drives the machine: 9 GiB < the 10 GiB bar → evict.
    try t.expectEqual(@as(Action, .evict), w.tick(0, avail));

    // A missing probe file is UNKNOWN, not zero: the guard idles (no action)
    // and says so once instead of flapping on every five-second wake.
    const missing_tmp = try std.fmt.allocPrint(t.allocator, "{s}/.zig-cache/tmp/{s}/absent", .{ cwd, tmp.sub_path });
    defer t.allocator.free(missing_tmp);
    const missing = try t.allocator.dupeSentinel(u8, missing_tmp, 0);
    defer t.allocator.free(missing);
    var w2 = PressureWatch{ .watermark_bytes = 10 * GBf, .probe_file = missing };
    try t.expectEqual(std.math.maxInt(u64), w2.availableBytes(io, t.allocator));
    try t.expectEqual(@as(Action, .none), w2.tick(0, w2.availableBytes(io, t.allocator)));

    const garbage_tmp = try std.fmt.allocPrint(t.allocator, "{s}/.zig-cache/tmp/{s}/junk", .{ cwd, tmp.sub_path });
    defer t.allocator.free(garbage_tmp);
    const garbage = try t.allocator.dupeSentinel(u8, garbage_tmp, 0);
    defer t.allocator.free(garbage);
    try tmp.dir.writeFile(io, .{ .sub_path = "junk", .data = "not-a-number" });
    var w3 = PressureWatch{ .watermark_bytes = 10 * GBf, .probe_file = garbage };
    try t.expectEqual(std.math.maxInt(u64), w3.availableBytes(io, t.allocator));
}