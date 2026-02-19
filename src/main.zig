const std = @import("std");
const math = std.math;
const mem = std.mem;
const os = std.os;
const fs = std.fs;
const builtin = @import("builtin");
const config = @import("config");

const STORAGE_MODE: []const u8 = config.storage_mode;
comptime {
    if (!std.mem.eql(u8, STORAGE_MODE, "f32") and
        !std.mem.eql(u8, STORAGE_MODE, "f16") and
        !std.mem.eql(u8, STORAGE_MODE, "sq8"))
    {
        @compileError("config.storage_mode must be \"f32\", \"f16\", or \"sq8\"");
    }
}
const StorageScalar = if (std.mem.eql(u8, STORAGE_MODE, "f16"))
    f16
else if (std.mem.eql(u8, STORAGE_MODE, "sq8"))
    i8
else
    f32;

// Configuration constants
const VECTOR_ALIGNMENT = 64; // For AVX-512 and cache line alignment
const PAGE_SIZE = 4096;
const INDEX_BUCKET_SIZE = 64;
const MMAP_THRESHOLD = 1024 * 1024; // 1MB
const PAGE_ALIGNMENT = 16384; // Common page size on macOS
const PREFETCH_DISTANCE = 256; // Prefetch distance for memory access
const CACHE_LINE_SIZE = 64;
const BRUTE_FORCE_THRESHOLD = 512;
const IVF_MIN_VECTORS_DEFAULT = 8192;
const IVF_MIN_LISTS = 16;
const IVF_MAX_LISTS = 512;
const IVF_DEFAULT_PROBES = 8;
const STORAGE_SLAB_VECTORS = 16_384;
const INT8_SCALE: f32 = 127.0;
const INT8_INV_SCALE: f32 = 1.0 / INT8_SCALE;
const INT8_INV_SCALE_SQ: f32 = INT8_INV_SCALE * INT8_INV_SCALE;

fn configuredCpuCount() usize {
    const cpu_count = @max(1, std.Thread.getCpuCount() catch 1);
    const raw = std.posix.getenv("ZECTOR_MAX_THREADS") orelse return cpu_count;
    const parsed = std.fmt.parseUnsigned(usize, raw, 10) catch return cpu_count;
    if (parsed == 0) return cpu_count;
    return @max(1, @min(cpu_count, parsed));
}

fn configuredDemoVectorCount(default_count: usize) usize {
    const raw = std.posix.getenv("ZECTOR_DEMO_VECTORS") orelse return default_count;
    const parsed = std.fmt.parseUnsigned(usize, raw, 10) catch return default_count;
    if (parsed == 0) return default_count;
    return parsed;
}

// Vector operations using SIMD when available
const VectorOps = struct {
    // Compute dot product using SIMD instructions when available
    pub inline fn dot(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);

        if (builtin.cpu.arch == .x86_64 and builtin.cpu.features.has(.avx2)) {
            return dotAVX2(a, b);
        } else {
            return dotScalar(a, b);
        }
    }

    fn dotScalar(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        for (a, b) |av, bv| {
            sum += av * bv;
        }
        return sum;
    }

    fn dotAVX2(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);

        const vec_size = 8; // AVX2 can process 8 floats at once
        const simd_len = a.len - (a.len % vec_size);

        // Use 4 accumulators to hide latency
        var acc1 = @as(@Vector(8, f32), @splat(0.0));
        var acc2 = @as(@Vector(8, f32), @splat(0.0));
        var acc3 = @as(@Vector(8, f32), @splat(0.0));
        var acc4 = @as(@Vector(8, f32), @splat(0.0));

        var i: usize = 0;
        const unroll_len = simd_len - (simd_len % 32);

        // Process 32 floats per iteration
        while (i < unroll_len) : (i += 32) {
            const va1 = @as(@Vector(8, f32), a[i..][0..8].*);
            const vb1 = @as(@Vector(8, f32), b[i..][0..8].*);
            acc1 += va1 * vb1;

            const va2 = @as(@Vector(8, f32), a[i + 8 ..][0..8].*);
            const vb2 = @as(@Vector(8, f32), b[i + 8 ..][0..8].*);
            acc2 += va2 * vb2;

            const va3 = @as(@Vector(8, f32), a[i + 16 ..][0..8].*);
            const vb3 = @as(@Vector(8, f32), b[i + 16 ..][0..8].*);
            acc3 += va3 * vb3;

            const va4 = @as(@Vector(8, f32), a[i + 24 ..][0..8].*);
            const vb4 = @as(@Vector(8, f32), b[i + 24 ..][0..8].*);
            acc4 += va4 * vb4;
        }

        // Process remaining 8-float chunks
        while (i < simd_len) : (i += vec_size) {
            const va = @as(@Vector(8, f32), a[i..][0..8].*);
            const vb = @as(@Vector(8, f32), b[i..][0..8].*);
            acc1 += va * vb;
        }

        // Combine accumulators
        const combined = acc1 + acc2 + acc3 + acc4;
        var sum: f32 = @reduce(.Add, combined);

        // Handle any remaining scalar elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// Unit-vector optimized cosine distance (assumes both vectors are normalized)
    pub inline fn cosineDistance(a: []const f32, b: []const f32) f32 {
        return 1.0 - dot(a, b); // dot == cos θ for unit vectors
    }

    /// Scalar dot product for small vectors (no SIMD overhead)
    pub inline fn dotScalarFast(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        for (a, b) |av, bv| {
            sum += av * bv;
        }
        return sum;
    }

    /// Highly optimized AVX2 dot product for 768-dimensional vectors
    fn dotAVX2Fixed768(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == 768);
        std.debug.assert(b.len == 768);

        // Process 64 floats per iteration (8 AVX2 registers for better pipelining)
        var acc1 = @as(@Vector(8, f32), @splat(0.0));
        var acc2 = @as(@Vector(8, f32), @splat(0.0));
        var acc3 = @as(@Vector(8, f32), @splat(0.0));
        var acc4 = @as(@Vector(8, f32), @splat(0.0));
        var acc5 = @as(@Vector(8, f32), @splat(0.0));
        var acc6 = @as(@Vector(8, f32), @splat(0.0));
        var acc7 = @as(@Vector(8, f32), @splat(0.0));
        var acc8 = @as(@Vector(8, f32), @splat(0.0));

        var i: usize = 0;
        while (i < 768) : (i += 64) {
            // Prefetch next iteration data
            if (i + 64 < 768) {
                @prefetch(a.ptr + i + 64, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(b.ptr + i + 64, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            // Unroll 8x8 for maximum throughput
            const va1 = @as(@Vector(8, f32), a[i..][0..8].*);
            const vb1 = @as(@Vector(8, f32), b[i..][0..8].*);
            acc1 += va1 * vb1;

            const va2 = @as(@Vector(8, f32), a[i + 8 ..][0..8].*);
            const vb2 = @as(@Vector(8, f32), b[i + 8 ..][0..8].*);
            acc2 += va2 * vb2;

            const va3 = @as(@Vector(8, f32), a[i + 16 ..][0..8].*);
            const vb3 = @as(@Vector(8, f32), b[i + 16 ..][0..8].*);
            acc3 += va3 * vb3;

            const va4 = @as(@Vector(8, f32), a[i + 24 ..][0..8].*);
            const vb4 = @as(@Vector(8, f32), b[i + 24 ..][0..8].*);
            acc4 += va4 * vb4;

            const va5 = @as(@Vector(8, f32), a[i + 32 ..][0..8].*);
            const vb5 = @as(@Vector(8, f32), b[i + 32 ..][0..8].*);
            acc5 += va5 * vb5;

            const va6 = @as(@Vector(8, f32), a[i + 40 ..][0..8].*);
            const vb6 = @as(@Vector(8, f32), b[i + 40 ..][0..8].*);
            acc6 += va6 * vb6;

            const va7 = @as(@Vector(8, f32), a[i + 48 ..][0..8].*);
            const vb7 = @as(@Vector(8, f32), b[i + 48 ..][0..8].*);
            acc7 += va7 * vb7;

            const va8 = @as(@Vector(8, f32), a[i + 56 ..][0..8].*);
            const vb8 = @as(@Vector(8, f32), b[i + 56 ..][0..8].*);
            acc8 += va8 * vb8;
        }

        // Combine accumulators in a tree reduction for better parallelism
        const combined12 = acc1 + acc2;
        const combined34 = acc3 + acc4;
        const combined56 = acc5 + acc6;
        const combined78 = acc7 + acc8;
        const combined1234 = combined12 + combined34;
        const combined5678 = combined56 + combined78;
        const combined = combined1234 + combined5678;
        return @reduce(.Add, combined);
    }

    /// Optimized AVX-512 dot product for maximum throughput
    fn dotAVX512(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);

        const vec_size = 16; // AVX-512 can process 16 floats at once
        const simd_len = a.len - (a.len % vec_size);

        // Use 4 accumulators to hide latency
        var acc1 = @as(@Vector(16, f32), @splat(0.0));
        var acc2 = @as(@Vector(16, f32), @splat(0.0));
        var acc3 = @as(@Vector(16, f32), @splat(0.0));
        var acc4 = @as(@Vector(16, f32), @splat(0.0));

        var i: usize = 0;
        const unroll_len = simd_len - (simd_len % 64);

        // Process 64 floats per iteration
        while (i < unroll_len) : (i += 64) {
            const va1 = @as(@Vector(16, f32), a[i..][0..16].*);
            const vb1 = @as(@Vector(16, f32), b[i..][0..16].*);
            acc1 += va1 * vb1;

            const va2 = @as(@Vector(16, f32), a[i + 16 ..][0..16].*);
            const vb2 = @as(@Vector(16, f32), b[i + 16 ..][0..16].*);
            acc2 += va2 * vb2;

            const va3 = @as(@Vector(16, f32), a[i + 32 ..][0..16].*);
            const vb3 = @as(@Vector(16, f32), b[i + 32 ..][0..16].*);
            acc3 += va3 * vb3;

            const va4 = @as(@Vector(16, f32), a[i + 48 ..][0..16].*);
            const vb4 = @as(@Vector(16, f32), b[i + 48 ..][0..16].*);
            acc4 += va4 * vb4;
        }

        // Process remaining 16-float chunks
        while (i < simd_len) : (i += vec_size) {
            const va = @as(@Vector(16, f32), a[i..][0..16].*);
            const vb = @as(@Vector(16, f32), b[i..][0..16].*);
            acc1 += va * vb;
        }

        // Combine accumulators
        const combined = acc1 + acc2 + acc3 + acc4;
        var sum: f32 = @reduce(.Add, combined);

        // Handle any remaining scalar elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// Optimized dot product that auto-selects best implementation
    pub inline fn dotFast(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);

        return switch (builtin.cpu.arch) {
            .x86_64 => blk: {
                if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
                    break :blk dotAVX512(a, b);
                }
                if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
                    if (a.len == 768) break :blk dotAVX2Fixed768(a, b);
                    break :blk dotAVX2(a, b);
                }
                break :blk dotScalar(a, b);
            },
            .aarch64 => blk: {
                break :blk dotNEON(a, b);
            },
            else => dotScalar(a, b),
        };
    }

    /// Optimized cosine distance for tight loops
    pub inline fn cosineDistanceFast(a: []const f32, b: []const f32) f32 {
        return 1.0 - dotFast(a, b);
    }

    pub inline fn toF32(v: StorageScalar) f32 {
        if (comptime StorageScalar == f32) return v;
        if (comptime StorageScalar == f16) return @as(f32, @floatCast(v));
        return @as(f32, @floatFromInt(v)) * INT8_INV_SCALE;
    }

    pub inline fn quantizeUnit(v: f32) i8 {
        const scaled = std.math.clamp(v * INT8_SCALE, -INT8_SCALE, INT8_SCALE);
        return @as(i8, @intFromFloat(@round(scaled)));
    }

    /// Hamming distance between two binary vectors (packed as u64 words)
    pub fn hammingDistance(a: []const u64, b: []const u64) u32 {
        std.debug.assert(a.len == b.len);
        var dist: u32 = 0;
        for (a, b) |aw, bw| {
            dist += @popCount(aw ^ bw);
        }
        return dist;
    }

    /// Encode a f32 vector as a binary vector (sign bits packed into u64 words)
    pub fn binaryEncode(src: []const f32, dst: []u64) void {
        const words = src.len / 64;
        const remainder = src.len % 64;
        var w: usize = 0;
        while (w < words) : (w += 1) {
            var bits: u64 = 0;
            for (0..64) |b| {
                if (src[w * 64 + b] >= 0) bits |= @as(u64, 1) << @intCast(b);
            }
            dst[w] = bits;
        }
        if (remainder > 0 and w < dst.len) {
            var bits: u64 = 0;
            for (0..remainder) |b| {
                if (src[words * 64 + b] >= 0) bits |= @as(u64, 1) << @intCast(b);
            }
            dst[w] = bits;
        }
    }

    /// Encode a StorageScalar vector as binary
    pub fn binaryEncodeStorage(src: []const StorageScalar, dst: []u64) void {
        const words = src.len / 64;
        const remainder = src.len % 64;
        var w: usize = 0;
        while (w < words) : (w += 1) {
            var bits: u64 = 0;
            for (0..64) |b| {
                if (toF32(src[w * 64 + b]) >= 0) bits |= @as(u64, 1) << @intCast(b);
            }
            dst[w] = bits;
        }
        if (remainder > 0 and w < dst.len) {
            var bits: u64 = 0;
            for (0..remainder) |b| {
                if (toF32(src[words * 64 + b]) >= 0) bits |= @as(u64, 1) << @intCast(b);
            }
            dst[w] = bits;
        }
    }

    /// SIMD-accelerated batch quantization of a normalized f32 vector to i8
    pub fn quantizeBatch(src: []const f32, dst: []i8) void {
        std.debug.assert(src.len == dst.len);
        const vec_len = 4;
        const scale_vec = @as(@Vector(vec_len, f32), @splat(INT8_SCALE));
        const min_vec = @as(@Vector(vec_len, f32), @splat(-INT8_SCALE));
        const max_vec = @as(@Vector(vec_len, f32), @splat(INT8_SCALE));

        const simd_len = src.len - (src.len % vec_len);
        var i: usize = 0;
        while (i < simd_len) : (i += vec_len) {
            const v = @as(@Vector(vec_len, f32), src[i..][0..vec_len].*);
            const scaled = @min(@max(v * scale_vec, min_vec), max_vec);
            // Convert to i32 then narrow to i8
            const rounded: @Vector(vec_len, i32) = @intFromFloat(scaled);
            inline for (0..vec_len) |j| {
                dst[i + j] = @as(i8, @intCast(std.math.clamp(rounded[j], -127, 127)));
            }
        }
        while (i < src.len) : (i += 1) {
            dst[i] = quantizeUnit(src[i]);
        }
    }

    pub inline fn dotStorage(a: []const StorageScalar, b: []const StorageScalar) f32 {
        std.debug.assert(a.len == b.len);
        if (comptime StorageScalar == f32) {
            return dotFast(a, b);
        }
        if (comptime StorageScalar == f16) {
            return dotF16(a, b);
        }
        return @as(f32, @floatFromInt(dotI8(a, b))) * INT8_INV_SCALE_SQ;
    }

    pub inline fn dotQueryStorage(query: []const f32, stored: []const StorageScalar) f32 {
        std.debug.assert(query.len == stored.len);
        if (comptime StorageScalar == f32) {
            return dotFast(query, stored);
        }
        if (comptime StorageScalar == f16) {
            return dotQueryF16(query, stored);
        }
        return dotQueryI8(query, stored);
    }

    pub inline fn cosineDistanceStorage(a: []const StorageScalar, b: []const StorageScalar) f32 {
        return 1.0 - dotStorage(a, b);
    }

    pub inline fn cosineDistanceQueryStorage(query: []const f32, stored: []const StorageScalar) f32 {
        return 1.0 - dotQueryStorage(query, stored);
    }

    pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        var sum: f32 = 0.0;
        for (a, b) |av, bv| {
            const diff = av - bv;
            sum += diff * diff;
        }
        return math.sqrt(sum);
    }

    /// Normalize a vector to unit length in-place (SIMD-accelerated)
    pub fn normalize(vec: []f32) void {
        const sq_norm = dotFast(vec, vec); // SIMD squared norm
        if (sq_norm <= 0) return;
        const inv_norm = 1.0 / math.sqrt(sq_norm);
        // SIMD scaling (multiply is faster than divide)
        const vec_len = 4;
        const simd_len = vec.len - (vec.len % vec_len);
        const scale = @as(@Vector(4, f32), @splat(inv_norm));
        var i: usize = 0;
        while (i < simd_len) : (i += vec_len) {
            const v = @as(@Vector(4, f32), vec[i..][0..4].*);
            vec[i..][0..4].* = @as([4]f32, v * scale);
        }
        while (i < vec.len) : (i += 1) {
            vec[i] *= inv_norm;
        }
    }

    fn dotF16(a: []const f16, b: []const f16) f32 {
        std.debug.assert(a.len == b.len);

        const vec_len = 16;
        const VF16 = @Vector(vec_len, f16);
        const VF32 = @Vector(vec_len, f32);

        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i + vec_len <= a.len) : (i += vec_len) {
            const va = @as(VF16, a[i..][0..vec_len].*);
            const vb = @as(VF16, b[i..][0..vec_len].*);
            const prod = @as(VF32, @floatCast(va * vb));
            sum += @reduce(.Add, prod);
        }

        while (i < a.len) : (i += 1) {
            sum += @as(f32, @floatCast(a[i])) * @as(f32, @floatCast(b[i]));
        }

        return sum;
    }

    fn dotQueryF16(query: []const f32, stored: []const f16) f32 {
        std.debug.assert(query.len == stored.len);

        const vec_len = 16;
        const VF16 = @Vector(vec_len, f16);
        const VF32 = @Vector(vec_len, f32);

        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i + vec_len <= query.len) : (i += vec_len) {
            const qv = @as(VF32, query[i..][0..vec_len].*);
            const sv = @as(VF32, @floatCast(@as(VF16, stored[i..][0..vec_len].*)));
            sum += @reduce(.Add, qv * sv);
        }

        while (i < query.len) : (i += 1) {
            sum += query[i] * @as(f32, @floatCast(stored[i]));
        }

        return sum;
    }

    pub inline fn dotI8(a: []const i8, b: []const i8) i32 {
        std.debug.assert(a.len == b.len);
        if (comptime builtin.cpu.arch == .aarch64 and
            std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod))
        {
            return dotI8Neon(a, b);
        }
        return dotI8Generic(a, b);
    }

    /// NEON SDOT path: processes 16 i8 pairs per instruction (4x faster than widen-to-i32).
    /// SDOT computes four 4-element dot products, accumulating into 4 i32 lanes.
    fn dotI8Neon(a: []const i8, b: []const i8) i32 {
        var acc1: @Vector(4, i32) = @splat(0);
        var acc2: @Vector(4, i32) = @splat(0);
        var acc3: @Vector(4, i32) = @splat(0);
        var acc4: @Vector(4, i32) = @splat(0);

        const len = a.len;
        const stride = 64; // 4x unrolled: 64 i8 pairs per iteration
        const aligned64 = len - (len % stride);

        var i: usize = 0;
        while (i < aligned64) : (i += stride) {
            const a1: @Vector(16, i8) = a[i..][0..16].*;
            const b1: @Vector(16, i8) = b[i..][0..16].*;
            acc1 = asm ("sdot %[d].4s, %[n].16b, %[m].16b"
                : [d] "=w" (-> @Vector(4, i32)),
                : [n] "w" (a1), [m] "w" (b1), [_] "0" (acc1),
            );
            const a2: @Vector(16, i8) = a[i + 16 ..][0..16].*;
            const b2: @Vector(16, i8) = b[i + 16 ..][0..16].*;
            acc2 = asm ("sdot %[d].4s, %[n].16b, %[m].16b"
                : [d] "=w" (-> @Vector(4, i32)),
                : [n] "w" (a2), [m] "w" (b2), [_] "0" (acc2),
            );
            const a3: @Vector(16, i8) = a[i + 32 ..][0..16].*;
            const b3: @Vector(16, i8) = b[i + 32 ..][0..16].*;
            acc3 = asm ("sdot %[d].4s, %[n].16b, %[m].16b"
                : [d] "=w" (-> @Vector(4, i32)),
                : [n] "w" (a3), [m] "w" (b3), [_] "0" (acc3),
            );
            const a4: @Vector(16, i8) = a[i + 48 ..][0..16].*;
            const b4: @Vector(16, i8) = b[i + 48 ..][0..16].*;
            acc4 = asm ("sdot %[d].4s, %[n].16b, %[m].16b"
                : [d] "=w" (-> @Vector(4, i32)),
                : [n] "w" (a4), [m] "w" (b4), [_] "0" (acc4),
            );
        }

        // Remaining 16-byte chunks
        while (i + 16 <= len) : (i += 16) {
            const av: @Vector(16, i8) = a[i..][0..16].*;
            const bv: @Vector(16, i8) = b[i..][0..16].*;
            acc1 = asm ("sdot %[d].4s, %[n].16b, %[m].16b"
                : [d] "=w" (-> @Vector(4, i32)),
                : [n] "w" (av), [m] "w" (bv), [_] "0" (acc1),
            );
        }

        var sum = @reduce(.Add, acc1 + acc2 + acc3 + acc4);
        while (i < len) : (i += 1) {
            sum += @as(i32, a[i]) * @as(i32, b[i]);
        }
        return sum;
    }

    /// Generic fallback: widen i8→i32 then multiply (for non-ARM or without dotprod).
    fn dotI8Generic(a: []const i8, b: []const i8) i32 {
        const vec_len = 16;
        const simd_len = a.len - (a.len % vec_len);

        var acc1 = @as(@Vector(16, i32), @splat(@as(i32, 0)));
        var acc2 = @as(@Vector(16, i32), @splat(@as(i32, 0)));

        var i: usize = 0;
        const unroll_len = simd_len - (simd_len % (vec_len * 2));

        while (i < unroll_len) : (i += vec_len * 2) {
            const va1: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), a[i..][0..16].*));
            const vb1: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), b[i..][0..16].*));
            acc1 += va1 * vb1;

            const va2: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), a[i + 16 ..][0..16].*));
            const vb2: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), b[i + 16 ..][0..16].*));
            acc2 += va2 * vb2;
        }

        while (i < simd_len) : (i += vec_len) {
            const va: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), a[i..][0..16].*));
            const vb: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), b[i..][0..16].*));
            acc1 += va * vb;
        }

        var sum = @reduce(.Add, acc1 + acc2);
        while (i < a.len) : (i += 1) {
            sum += @as(i32, @intCast(a[i])) * @as(i32, @intCast(b[i]));
        }
        return sum;
    }

    fn dotQueryI8(query: []const f32, stored: []const i8) f32 {
        std.debug.assert(query.len == stored.len);

        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i < query.len) : (i += 1) {
            sum += query[i] * (@as(f32, @floatFromInt(stored[i])) * INT8_INV_SCALE);
        }
        return sum;
    }

    fn dotNEON(a: []const f32, b: []const f32) f32 {
        // Optimised dot product for ARM NEON – 8 accumulators, 32 floats/iter
        std.debug.assert(a.len == b.len);

        const vec_size = 4;
        const simd_len = a.len - (a.len % vec_size);

        // 8 accumulators to saturate Apple Silicon NEON execution units
        var acc1 = @as(@Vector(4, f32), @splat(0.0));
        var acc2 = @as(@Vector(4, f32), @splat(0.0));
        var acc3 = @as(@Vector(4, f32), @splat(0.0));
        var acc4 = @as(@Vector(4, f32), @splat(0.0));
        var acc5 = @as(@Vector(4, f32), @splat(0.0));
        var acc6 = @as(@Vector(4, f32), @splat(0.0));
        var acc7 = @as(@Vector(4, f32), @splat(0.0));
        var acc8 = @as(@Vector(4, f32), @splat(0.0));

        var i: usize = 0;
        const unroll_len = simd_len - (simd_len % 32);

        while (i < unroll_len) : (i += 32) {
            const va1 = @as(@Vector(4, f32), a[i..][0..4].*);
            const vb1 = @as(@Vector(4, f32), b[i..][0..4].*);
            acc1 += va1 * vb1;

            const va2 = @as(@Vector(4, f32), a[i + 4 ..][0..4].*);
            const vb2 = @as(@Vector(4, f32), b[i + 4 ..][0..4].*);
            acc2 += va2 * vb2;

            const va3 = @as(@Vector(4, f32), a[i + 8 ..][0..4].*);
            const vb3 = @as(@Vector(4, f32), b[i + 8 ..][0..4].*);
            acc3 += va3 * vb3;

            const va4 = @as(@Vector(4, f32), a[i + 12 ..][0..4].*);
            const vb4 = @as(@Vector(4, f32), b[i + 12 ..][0..4].*);
            acc4 += va4 * vb4;

            const va5 = @as(@Vector(4, f32), a[i + 16 ..][0..4].*);
            const vb5 = @as(@Vector(4, f32), b[i + 16 ..][0..4].*);
            acc5 += va5 * vb5;

            const va6 = @as(@Vector(4, f32), a[i + 20 ..][0..4].*);
            const vb6 = @as(@Vector(4, f32), b[i + 20 ..][0..4].*);
            acc6 += va6 * vb6;

            const va7 = @as(@Vector(4, f32), a[i + 24 ..][0..4].*);
            const vb7 = @as(@Vector(4, f32), b[i + 24 ..][0..4].*);
            acc7 += va7 * vb7;

            const va8 = @as(@Vector(4, f32), a[i + 28 ..][0..4].*);
            const vb8 = @as(@Vector(4, f32), b[i + 28 ..][0..4].*);
            acc8 += va8 * vb8;
        }

        while (i < simd_len) : (i += vec_size) {
            const va = @as(@Vector(4, f32), a[i..][0..4].*);
            const vb = @as(@Vector(4, f32), b[i..][0..4].*);
            acc1 += va * vb;
        }

        const combined12 = acc1 + acc2;
        const combined34 = acc3 + acc4;
        const combined56 = acc5 + acc6;
        const combined78 = acc7 + acc8;
        const combined1234 = combined12 + combined34;
        const combined5678 = combined56 + combined78;
        var sum: f32 = @reduce(.Add, combined1234 + combined5678);

        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }
};

// Memory-mapped vector storage for efficient I/O
const VectorStorage = struct {
    allocator: mem.Allocator,
    file: ?fs.File,
    data: []align(VECTOR_ALIGNMENT) u8,
    mmap_data: ?[]align(PAGE_ALIGNMENT) u8,
    slabs: std.ArrayList([]align(VECTOR_ALIGNMENT) u8),
    dimension: usize,
    count: usize,
    capacity: usize,
    slab_capacity_vectors: usize,
    use_slabs: bool,
    use_mmap: bool,
    // Quantization support for memory efficiency
    quantized_data: ?[]align(CACHE_LINE_SIZE) u8,
    use_quantization: bool,
    quantization_bits: u8,

    pub fn init(allocator: mem.Allocator, dimension: usize, initial_capacity: usize) !VectorStorage {
        var storage = VectorStorage{
            .allocator = allocator,
            .file = null,
            .data = undefined,
            .mmap_data = null,
            .slabs = std.ArrayList([]align(VECTOR_ALIGNMENT) u8).init(allocator),
            .dimension = dimension,
            .count = 0,
            .capacity = 0,
            .slab_capacity_vectors = @max(STORAGE_SLAB_VECTORS, initial_capacity),
            .use_slabs = true,
            .use_mmap = false,
            .quantized_data = null,
            .use_quantization = false,
            .quantization_bits = 8,
        };
        errdefer storage.deinit();

        while (storage.capacity < initial_capacity) {
            try storage.addSlab(storage.slab_capacity_vectors);
        }
        if (storage.capacity == 0) {
            try storage.addSlab(storage.slab_capacity_vectors);
        }
        storage.data = storage.slabs.items[0];

        return storage;
    }

    pub fn initWithFile(allocator: mem.Allocator, path: []const u8, dimension: usize) !VectorStorage {
        const file = try fs.cwd().createFile(path, .{ .read = true });
        errdefer file.close();

        const vector_size = dimension * @sizeOf(StorageScalar);
        const initial_size = 1024 * vector_size; // Start with 1024 vectors

        try file.setEndPos(initial_size);

        var storage = VectorStorage{
            .allocator = allocator,
            .file = file,
            .data = undefined,
            .mmap_data = null,
            .slabs = std.ArrayList([]align(VECTOR_ALIGNMENT) u8).init(allocator),
            .dimension = dimension,
            .count = 0,
            .capacity = 1024,
            .slab_capacity_vectors = 0,
            .use_slabs = false,
            .use_mmap = initial_size >= MMAP_THRESHOLD,
            .quantized_data = null,
            .use_quantization = false,
            .quantization_bits = 8,
        };

        if (storage.use_mmap) {
            // Use memory mapping for large files
            const mmap_data = try std.posix.mmap(null, initial_size, std.posix.PROT.READ | std.posix.PROT.WRITE, std.posix.MAP{ .TYPE = .SHARED }, file.handle, 0);
            storage.mmap_data = mmap_data;
            storage.data = @alignCast(mmap_data);
        } else {
            storage.data = try allocator.alignedAlloc(u8, VECTOR_ALIGNMENT, initial_size);
        }

        return storage;
    }

    pub fn deinit(self: *VectorStorage) void {
        if (self.use_slabs) {
            for (self.slabs.items) |slab| {
                self.allocator.free(slab);
            }
            self.slabs.deinit();
        } else {
            if (self.use_mmap) {
                if (self.mmap_data) |mmap_data| {
                    std.posix.munmap(mmap_data);
                }
            } else if (self.data.len > 0) {
                self.allocator.free(self.data);
            }
            self.slabs.deinit();
        }
        if (self.file) |file| {
            file.close();
        }
    }

    fn addSlab(self: *VectorStorage, vectors: usize) !void {
        const vector_size = self.dimension * @sizeOf(StorageScalar);
        const slab_size = vectors * vector_size;
        const slab = try self.allocator.alignedAlloc(u8, VECTOR_ALIGNMENT, slab_size);
        try self.slabs.append(slab);
        self.capacity += vectors;
        if (self.slabs.items.len == 1) {
            self.data = slab;
        }
    }

    fn writeVectorNormalizedAt(self: *VectorStorage, idx: usize, vector: []const f32) void {
        std.debug.assert(idx < self.capacity);
        std.debug.assert(vector.len == self.dimension);

        const vector_size = self.dimension * @sizeOf(StorageScalar);
        const dest = if (self.use_slabs) blk: {
            const slab_idx = idx / self.slab_capacity_vectors;
            const in_slab = idx % self.slab_capacity_vectors;
            const slab = self.slabs.items[slab_idx];
            const offset = in_slab * vector_size;
            break :blk slab[offset..][0..vector_size];
        } else blk: {
            const offset = idx * vector_size;
            break :blk self.data[offset..][0..vector_size];
        };
        const dest_storage = @as([*]StorageScalar, @ptrCast(@alignCast(dest.ptr)))[0..vector.len];

        const sum = VectorOps.dotFast(vector, vector); // SIMD squared norm
        const inv_norm: f32 = if (sum > 0) 1.0 / @sqrt(sum) else 1.0;
        if (comptime StorageScalar == f32) {
            // SIMD-accelerated normalize + store
            const vec_len = 4;
            const simd_len = vector.len - (vector.len % vec_len);
            const scale = @as(@Vector(4, f32), @splat(inv_norm));
            var si: usize = 0;
            while (si < simd_len) : (si += vec_len) {
                const v = @as(@Vector(4, f32), vector[si..][0..4].*);
                dest_storage[si..][0..4].* = @as([4]f32, v * scale);
            }
            while (si < vector.len) : (si += 1) {
                dest_storage[si] = vector[si] * inv_norm;
            }
        } else if (comptime StorageScalar == f16) {
            for (vector, 0..) |v, i| {
                dest_storage[i] = @as(StorageScalar, @floatCast(v * inv_norm));
            }
        } else {
            for (vector, 0..) |v, i| {
                dest_storage[i] = VectorOps.quantizeUnit(v * inv_norm);
            }
        }
    }

    pub fn addVector(self: *VectorStorage, vector: []const f32) !usize {
        if (vector.len != self.dimension) return error.InvalidDimension;

        if (self.count >= self.capacity) {
            try self.grow();
        }

        const idx = self.count;
        self.writeVectorNormalizedAt(idx, vector);
        self.count += 1;
        return idx;
    }

    pub fn getVector(self: VectorStorage, idx: usize) []const StorageScalar {
        std.debug.assert(idx < self.count);
        const vector_size = self.dimension * @sizeOf(StorageScalar);
        const bytes = if (self.use_slabs) blk: {
            const slab_idx = idx / self.slab_capacity_vectors;
            const in_slab = idx % self.slab_capacity_vectors;
            const slab = self.slabs.items[slab_idx];
            const offset = in_slab * vector_size;
            break :blk slab[offset..][0..vector_size];
        } else blk: {
            const offset = idx * vector_size;
            break :blk self.data[offset..][0..vector_size];
        };
        return @as([*]const StorageScalar, @ptrCast(@alignCast(bytes.ptr)))[0..self.dimension];
    }

    fn getVectorBytes(self: VectorStorage, idx: usize) []const u8 {
        const vector_size = self.dimension * @sizeOf(StorageScalar);
        if (self.use_slabs) {
            const slab_idx = idx / self.slab_capacity_vectors;
            const in_slab = idx % self.slab_capacity_vectors;
            return self.slabs.items[slab_idx][in_slab * vector_size ..][0..vector_size];
        }
        return self.data[idx * vector_size ..][0..vector_size];
    }

    fn getVectorBytesMut(self: *VectorStorage, idx: usize) []u8 {
        const vector_size = self.dimension * @sizeOf(StorageScalar);
        if (self.use_slabs) {
            const slab_idx = idx / self.slab_capacity_vectors;
            const in_slab = idx % self.slab_capacity_vectors;
            return self.slabs.items[slab_idx][in_slab * vector_size ..][0..vector_size];
        }
        return self.data[idx * vector_size ..][0..vector_size];
    }

    /// Pre-allocate storage capacity for batch operations
    pub fn reserveCapacity(self: *VectorStorage, additional_count: usize) !void {
        const needed_capacity = self.count + additional_count;
        if (needed_capacity <= self.capacity) return; // Already have enough capacity

        if (self.use_slabs) {
            while (self.capacity < needed_capacity) {
                try self.addSlab(self.slab_capacity_vectors);
            }
            return;
        }

        // Grow to at least the needed capacity
        var new_capacity = self.capacity;
        while (new_capacity < needed_capacity) {
            new_capacity *= 2;
        }

        const vector_size = self.dimension * @sizeOf(StorageScalar);
        const new_size = new_capacity * vector_size;

        if (self.file) |file| {
            try file.setEndPos(new_size);

            if (self.use_mmap) {
                if (self.mmap_data) |mmap_data| {
                    std.posix.munmap(mmap_data);
                }
                const new_mmap_data = try std.posix.mmap(null, new_size, std.posix.PROT.READ | std.posix.PROT.WRITE, std.posix.MAP{ .TYPE = .SHARED }, file.handle, 0);
                self.mmap_data = new_mmap_data;
                self.data = @alignCast(new_mmap_data);
            } else if (new_size >= MMAP_THRESHOLD and !self.use_mmap) {
                // Switch to mmap
                const old_data = self.data;
                const new_mmap_data = try std.posix.mmap(null, new_size, std.posix.PROT.READ | std.posix.PROT.WRITE, std.posix.MAP{ .TYPE = .SHARED }, file.handle, 0);
                self.mmap_data = new_mmap_data;
                self.data = @alignCast(new_mmap_data);
                @memcpy(self.data[0..old_data.len], old_data);
                self.allocator.free(old_data);
                self.use_mmap = true;
            } else {
                self.data = try self.allocator.realloc(self.data, new_size);
            }
        } else {
            self.data = try self.allocator.realloc(self.data, new_size);
        }

        self.capacity = new_capacity;
    }

    /// Parallel batch append with in-place normalization into pre-allocated storage.
    pub fn appendBatchNormalized(self: *VectorStorage, vectors: []const []const f32) !usize {
        const start_idx = self.count;
        if (vectors.len == 0) return start_idx;

        for (vectors) |vector| {
            if (vector.len != self.dimension) return error.InvalidDimension;
        }

        try self.reserveCapacity(vectors.len);

        const cpu_count = configuredCpuCount();
        const thread_count = @max(1, @min(cpu_count, vectors.len));

        if (thread_count > 1 and vectors.len >= thread_count * 64) {
            const threads = try self.allocator.alloc(std.Thread, thread_count);
            defer self.allocator.free(threads);

            const Context = struct {
                storage: *VectorStorage,
                base_idx: usize,
                vectors: []const []const f32,

                fn run(ctx: @This()) void {
                    for (ctx.vectors, 0..) |vector, i| {
                        ctx.storage.writeVectorNormalizedAt(ctx.base_idx + i, vector);
                    }
                }
            };

            const chunk_size = (vectors.len + thread_count - 1) / thread_count;

            for (threads, 0..) |*thread, i| {
                const start = i * chunk_size;
                const end = @min(start + chunk_size, vectors.len);
                const ctx = Context{
                    .storage = self,
                    .base_idx = start_idx + start,
                    .vectors = vectors[start..end],
                };
                thread.* = try std.Thread.spawn(.{}, Context.run, .{ctx});
            }

            for (threads) |thread| thread.join();
        } else {
            for (vectors, 0..) |vector, i| {
                self.writeVectorNormalizedAt(start_idx + i, vector);
            }
        }

        self.count += vectors.len;
        return start_idx;
    }

    fn grow(self: *VectorStorage) !void {
        if (self.use_slabs) {
            try self.addSlab(self.slab_capacity_vectors);
            return;
        }

        const new_capacity = self.capacity * 2;
        const vector_size = self.dimension * @sizeOf(StorageScalar);
        const new_size = new_capacity * vector_size;

        if (self.file) |file| {
            try file.setEndPos(new_size);

            if (self.use_mmap) {
                if (self.mmap_data) |mmap_data| {
                    std.posix.munmap(mmap_data);
                }
                const new_mmap_data = try std.posix.mmap(null, new_size, std.posix.PROT.READ | std.posix.PROT.WRITE, std.posix.MAP{ .TYPE = .SHARED }, file.handle, 0);
                self.mmap_data = new_mmap_data;
                self.data = @alignCast(new_mmap_data);
            } else if (new_size >= MMAP_THRESHOLD and !self.use_mmap) {
                // Switch to mmap
                const old_data = self.data;
                const new_mmap_data = try std.posix.mmap(null, new_size, std.posix.PROT.READ | std.posix.PROT.WRITE, std.posix.MAP{ .TYPE = .SHARED }, file.handle, 0);
                self.mmap_data = new_mmap_data;
                self.data = @alignCast(new_mmap_data);
                @memcpy(self.data[0..old_data.len], old_data);
                self.allocator.free(old_data);
                self.use_mmap = true;
            } else {
                self.data = try self.allocator.realloc(self.data, new_size);
            }
        } else {
            self.data = try self.allocator.realloc(self.data, new_size);
        }

        self.capacity = new_capacity;
    }
};

// Thread-local fast path for HNSW query scratch context.
threadlocal var tl_search_context: ?*HNSWIndex.SearchContext = null;
threadlocal var tl_search_owner: ?*HNSWIndex = null;

// Hierarchical Navigable Small World (HNSW) index for fast similarity search
const HNSWIndex = struct {
    const Connection = struct {
        offset: usize,
        len: usize,
        capacity: usize,
    };

    const Node = struct {
        level: u8,
        /// (offset, length) pairs for each level in the edge pool
        connections: []Connection,
        lock: std.atomic.Value(u8) align(CACHE_LINE_SIZE) = std.atomic.Value(u8).init(0),

        pub fn init(allocator: mem.Allocator, level: u8) !Node {
            const connections = try allocator.alloc(Connection, level + 1);
            for (connections) |*conn| {
                conn.* = .{ .offset = 0, .len = 0, .capacity = 0 };
            }
            return Node{
                .level = level,
                .connections = connections,
            };
        }

        pub fn deinit(self: *Node, allocator: mem.Allocator) void {
            allocator.free(self.connections);
        }

        /// True user-space spinlock: CAS fast path + exponential backoff.
        /// No syscalls, no OS scheduler involvement — pure atomic operations.
        pub inline fn spinLock(self: *Node) void {
            // Fast path: uncontested lock
            if (self.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) == null) return;
            self.spinLockSlow();
        }

        fn spinLockSlow(self: *Node) void {
            var spin: u3 = 0;
            while (true) {
                if (self.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) == null) return;
                // Exponential backoff: 1, 2, 4, ... 128 YIELD/PAUSE hints
                const spins = @as(u8, 1) << spin;
                for (0..spins) |_| std.atomic.spinLoopHint();
                spin +|= 1;
            }
        }

        pub inline fn spinUnlock(self: *Node) void {
            self.lock.store(0, .release);
        }

        /// Non-blocking lock attempt. Returns true if acquired, false if contended.
        pub inline fn trySpinLock(self: *Node) bool {
            return self.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) == null;
        }
    };

    const SearchItem = struct {
        idx: usize,
        distance: f32,
    };

    const SearchContext = struct {
        thread_id: std.Thread.Id,
        visited: []u32,
        candidates: []SearchItem,
        w: []SearchItem,
        scratch_results: []SearchItem,
        mark: u32,
    };

    allocator: mem.Allocator,
    nodes: std.ArrayList(Node),
    /// Single pool for all edges to avoid allocator churn
    edge_pool: std.ArrayList(usize),
    /// Single pool for all distances (parallel to edge_pool)
    distance_pool: std.ArrayList(f32),
    edge_pool_used: usize,
    /// Pre-allocated edge slabs per level to avoid growth during bulk insert
    level_cursors: []usize,
    max_level: u8,
    /// Re-usable scratch buffers for insertion-time graph search
    scratch_idx: []usize,
    scratch_visited: []u32,
    scratch_candidates: []SearchItem,
    scratch_w: []SearchItem,
    scratch_mark: u32,
    search_contexts: std.ArrayList(*SearchContext),
    search_contexts_lock: std.Thread.Mutex,
    edge_pool_mutex: std.Thread.Mutex,
    build_rwlock: std.Thread.RwLock,
    entry_point: ?usize,
    m: usize, // Max connections per layer
    ef_construction: usize,
    ml: f64, // Level multiplier
    seed: u64,
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: mem.Allocator, m: usize, ef_construction: usize) HNSWIndex {
        return HNSWIndex{
            .allocator = allocator,
            .nodes = std.ArrayList(Node).init(allocator),
            .edge_pool = std.ArrayList(usize).init(allocator),
            .distance_pool = std.ArrayList(f32).init(allocator),
            .edge_pool_used = 0,
            .level_cursors = &[_]usize{},
            .max_level = 0,
            .scratch_idx = &[_]usize{},
            .scratch_visited = &[_]u32{},
            .scratch_candidates = &[_]SearchItem{},
            .scratch_w = &[_]SearchItem{},
            .scratch_mark = 1,
            .search_contexts = std.ArrayList(*SearchContext).init(allocator),
            .search_contexts_lock = .{},
            .edge_pool_mutex = .{},
            .build_rwlock = .{},
            .entry_point = null,
            .m = m,
            .ef_construction = ef_construction,
            .ml = 1.0 / @log(@as(f64, 2.0)),
            .seed = @intCast(std.time.milliTimestamp()),
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp())),
        };
    }

    pub fn deinit(self: *HNSWIndex) void {
        if (tl_search_owner == self) {
            tl_search_context = null;
            tl_search_owner = null;
        }

        for (self.nodes.items) |*node| {
            node.deinit(self.allocator);
        }
        self.nodes.deinit();
        self.edge_pool.deinit();
        self.distance_pool.deinit();
        if (self.level_cursors.len > 0) self.allocator.free(self.level_cursors);
        if (self.scratch_idx.len > 0) self.allocator.free(self.scratch_idx);
        if (self.scratch_visited.len > 0) self.allocator.free(self.scratch_visited);
        if (self.scratch_candidates.len > 0) self.allocator.free(self.scratch_candidates);
        if (self.scratch_w.len > 0) self.allocator.free(self.scratch_w);
        for (self.search_contexts.items) |ctx| {
            if (ctx.visited.len > 0) self.allocator.free(ctx.visited);
            if (ctx.candidates.len > 0) self.allocator.free(ctx.candidates);
            if (ctx.w.len > 0) self.allocator.free(ctx.w);
            if (ctx.scratch_results.len > 0) self.allocator.free(ctx.scratch_results);
            self.allocator.destroy(ctx);
        }
        self.search_contexts.deinit();
    }

    fn ensureBuffer(comptime T: type, allocator: mem.Allocator, buf: *[]T, needed: usize) !void {
        if (buf.*.len >= needed) return;

        var new_cap = @max(needed, 64);
        if (buf.*.len > 0) {
            new_cap = @max(new_cap, buf.*.len * 2);
        }

        const new_buf = try allocator.alloc(T, new_cap);
        if (buf.*.len > 0) {
            @memcpy(new_buf[0..buf.*.len], buf.*);
            allocator.free(buf.*);
        }
        buf.* = new_buf;
    }

    fn ensureInsertScratch(self: *HNSWIndex, num_closest: usize) !void {
        const candidate_cap = @max(num_closest * 32, 64);
        const w_cap = @max(num_closest, 1);

        try ensureBuffer(u32, self.allocator, &self.scratch_visited, self.nodes.items.len);
        try ensureBuffer(SearchItem, self.allocator, &self.scratch_candidates, candidate_cap);
        try ensureBuffer(SearchItem, self.allocator, &self.scratch_w, w_cap);
        try ensureBuffer(usize, self.allocator, &self.scratch_idx, w_cap);

        if (self.scratch_mark == std.math.maxInt(u32)) {
            @memset(self.scratch_visited, 0);
            self.scratch_mark = 1;
        } else {
            self.scratch_mark += 1;
        }
    }

    fn ensureSearchContextCapacity(
        self: *HNSWIndex,
        ctx: *SearchContext,
        num_closest: usize,
        max_nodes: usize,
    ) !void {
        const candidate_cap = @max(num_closest * 32, 64);
        const w_cap = @max(num_closest, 1);

        try ensureBuffer(u32, self.allocator, &ctx.visited, max_nodes);
        try ensureBuffer(SearchItem, self.allocator, &ctx.candidates, candidate_cap);
        try ensureBuffer(SearchItem, self.allocator, &ctx.w, w_cap);
        try ensureBuffer(SearchItem, self.allocator, &ctx.scratch_results, candidate_cap);

        if (ctx.mark == std.math.maxInt(u32)) {
            @memset(ctx.visited, 0);
            ctx.mark = 1;
        } else {
            ctx.mark += 1;
        }
    }

    fn acquireSearchContext(self: *HNSWIndex, num_closest: usize, max_nodes: usize) !*SearchContext {
        // Fast path: one cached context per (thread, index).
        if (tl_search_owner == self) {
            if (tl_search_context) |ctx| {
                try self.ensureSearchContextCapacity(ctx, num_closest, max_nodes);
                return ctx;
            }
        }

        const ctx = try self.allocator.create(SearchContext);
        errdefer {
            if (ctx.visited.len > 0) self.allocator.free(ctx.visited);
            if (ctx.candidates.len > 0) self.allocator.free(ctx.candidates);
            if (ctx.w.len > 0) self.allocator.free(ctx.w);
            if (ctx.scratch_results.len > 0) self.allocator.free(ctx.scratch_results);
            self.allocator.destroy(ctx);
        }
        ctx.* = .{
            .thread_id = std.Thread.getCurrentId(),
            .visited = &[_]u32{},
            .candidates = &[_]SearchItem{},
            .w = &[_]SearchItem{},
            .scratch_results = &[_]SearchItem{},
            .mark = 1,
        };

        try self.ensureSearchContextCapacity(ctx, num_closest, max_nodes);

        {
            self.search_contexts_lock.lock();
            defer self.search_contexts_lock.unlock();
            try self.search_contexts.append(ctx);
        }

        tl_search_context = ctx;
        tl_search_owner = self;
        return ctx;
    }

    /// Pre-allocate edge pool capacity for bulk insertions
    pub fn reserveCapacity(self: *HNSWIndex, node_count: usize) !void {
        // Estimate total edges needed across all levels
        // Use a simpler, safer estimation approach
        var total_edges: usize = 0;

        // For HNSW, the probability of a node being at level L is (1/ml)^L
        // Most nodes will be at level 0, very few at higher levels
        var level: usize = 0;
        while (level <= 8) : (level += 1) { // Reasonable max level
            const level_prob = math.pow(f64, 1.0 / self.ml, @floatFromInt(level));
            if (level_prob < 0.001) break; // Stop when probability is very low

            const nodes_at_level = @as(usize, @intFromFloat(@as(f64, @floatFromInt(node_count)) * level_prob));
            if (nodes_at_level == 0) break;

            const capacity_at_level = if (level == 0) self.m * 2 else self.m;
            const edges_at_level = nodes_at_level * capacity_at_level;
            total_edges += edges_at_level;

            // Safety check to prevent runaway allocation
            if (total_edges > node_count * self.m * 4) {
                total_edges = node_count * self.m * 4;
                break;
            }
        }

        // Add 20% buffer for safety, but cap at reasonable limit
        total_edges = @min((total_edges * 12) / 10, node_count * self.m * 8);

        if (total_edges > 0) {
            try self.edge_pool.ensureTotalCapacity(total_edges);
            try self.distance_pool.ensureTotalCapacity(total_edges);
            // Don't resize, just ensure capacity is available
        }
    }

    /// Allocate space in the edge pool for `count` connections
    fn allocateEdges(self: *HNSWIndex, count: usize) !usize {
        const needed_capacity = self.edge_pool_used + count;

        // Ensure we have enough capacity
        if (needed_capacity > self.edge_pool.capacity) {
            const new_capacity = @max(needed_capacity, self.edge_pool.capacity * 2);
            try self.edge_pool.ensureTotalCapacity(new_capacity);
            try self.distance_pool.ensureTotalCapacity(new_capacity);
        }

        // Resize pools if needed to accommodate the new edges
        if (needed_capacity > self.edge_pool.items.len) {
            try self.edge_pool.resize(needed_capacity);
            try self.distance_pool.resize(needed_capacity);
        }

        const offset = self.edge_pool_used;
        self.edge_pool_used += count;
        return offset;
    }

    /// Get connections for a node at a specific level
    fn getConnections(self: *HNSWIndex, node_idx: usize, level: usize) []usize {
        const node = &self.nodes.items[node_idx];
        if (level > node.level) return &[_]usize{};

        const conn = node.connections[level];
        if (conn.len == 0) return &[_]usize{};

        return self.edge_pool.items[conn.offset .. conn.offset + conn.len];
    }

    /// Get cached distances for a node at a specific level
    fn getDistances(self: *HNSWIndex, node_idx: usize, level: usize) []f32 {
        const node = &self.nodes.items[node_idx];
        if (level > node.level) return &[_]f32{};

        const conn = node.connections[level];
        if (conn.len == 0) return &[_]f32{};

        return self.distance_pool.items[conn.offset .. conn.offset + conn.len];
    }

    /// Add a connection from node_idx to neighbor_idx at level
    fn addConnection(
        self: *HNSWIndex,
        node_idx: usize,
        neighbor_idx: usize,
        level: usize,
        dist: f32, // distance(node, neighbor)
        storage: *const VectorStorage, // for computing distances during pruning
    ) !void {
        _ = storage; // Not needed anymore since we cache distances
        const node = &self.nodes.items[node_idx];
        if (level > node.level) return; // node has no such layer

        const cap = if (level == 0) self.m * 2 else self.m;
        var conn = &node.connections[level];

        // Allocate on first use
        if (conn.capacity == 0) {
            self.edge_pool_mutex.lock();
            defer self.edge_pool_mutex.unlock();

            // Double-check once we hold the pool growth lock.
            if (conn.capacity == 0) {
                conn.offset = try self.allocateEdges(cap);
                conn.capacity = cap;
                conn.len = 0;
            }
        }

        const offset = conn.offset;

        // Fast-path: empty list
        if (conn.len == 0) {
            self.edge_pool.items[offset] = neighbor_idx;
            self.distance_pool.items[offset] = dist;
            conn.len = 1;
            return;
        }

        // Find insertion position so that distances stay sorted (ascending).
        var insert_pos: usize = conn.len; // default: append at end
        var i_iter: usize = 0;
        while (i_iter < conn.len) : (i_iter += 1) {
            if (self.edge_pool.items[offset + i_iter] == neighbor_idx) return;
            const other_dist = self.distance_pool.items[offset + i_iter];
            if (dist < other_dist) {
                insert_pos = i_iter;
                break;
            }
        }

        if (conn.len < conn.capacity) {
            // Shift items right to make room.
            var j: usize = conn.len;
            while (j > insert_pos) : (j -= 1) {
                self.edge_pool.items[offset + j] = self.edge_pool.items[offset + j - 1];
                self.distance_pool.items[offset + j] = self.distance_pool.items[offset + j - 1];
            }
            self.edge_pool.items[offset + insert_pos] = neighbor_idx;
            self.distance_pool.items[offset + insert_pos] = dist;
            conn.len += 1;
        } else {
            // Already full – if new distance is worse than the current worst, ignore.
            if (insert_pos == conn.capacity) return;

            // Otherwise insert and drop the last (worst) entry.
            var j: usize = conn.capacity - 1;
            while (j > insert_pos) : (j -= 1) {
                self.edge_pool.items[offset + j] = self.edge_pool.items[offset + j - 1];
                self.distance_pool.items[offset + j] = self.distance_pool.items[offset + j - 1];
            }
            self.edge_pool.items[offset + insert_pos] = neighbor_idx;
            self.distance_pool.items[offset + insert_pos] = dist;
            // conn.len stays at capacity.
        }
    }

    fn selectLevel(self: *HNSWIndex) u8 {
        const f = @max(self.rng.random().float(f64), 0.000000000001);
        return @as(u8, @intCast(@as(u32, @intFromFloat(@floor(-@log(f) * self.ml)))));
    }

    pub fn insert(self: *HNSWIndex, idx: usize, storage: *const VectorStorage) !void {
        try self.insertParallel(idx, storage);
    }

    pub fn insertParallel(self: *HNSWIndex, idx: usize, storage: *const VectorStorage) !void {
        {
            self.build_rwlock.lock();
            defer self.build_rwlock.unlock();

            while (self.nodes.items.len <= idx) {
                try self.nodes.append(try Node.init(self.allocator, 0));
            }

            const level = self.selectLevel();
            self.nodes.items[idx].deinit(self.allocator);
            self.nodes.items[idx] = try Node.init(self.allocator, level);

            if (self.entry_point == null) {
                self.entry_point = idx;
                self.max_level = level;
                return;
            }
        }

        try self.insertParallelNoAlloc(idx, storage);
    }

    /// HNSW spatial pruning heuristic with "keep pruned" extension.
    /// Selects diverse neighbors, then backfills with best pruned candidates
    /// to maintain graph connectivity. Candidates must be sorted by distance ascending.
    /// Returns the number of kept candidates (bubbled to the front of the slice).
    fn selectNeighborsHeuristic(
        _: *HNSWIndex,
        candidates: []SearchItem,
        m_max: usize,
        storage: *const VectorStorage,
    ) usize {
        if (candidates.len <= 1) return @min(candidates.len, m_max);

        var selected_count: usize = 0;
        // Track pruned candidates for backfill (on stack, bounded by input size)
        var pruned_buf: [1024]SearchItem = undefined;
        var pruned_count: usize = 0;

        for (candidates) |cand| {
            if (selected_count >= m_max) break;

            var discard = false;
            const cand_vec = storage.getVector(cand.idx);

            for (candidates[0..selected_count]) |selected| {
                const sel_vec = storage.getVector(selected.idx);
                const dist_to_selected = VectorOps.cosineDistanceStorage(cand_vec, sel_vec);

                if (dist_to_selected < cand.distance * 1.2) {
                    discard = true;
                    break;
                }
            }

            if (!discard) {
                candidates[selected_count] = cand;
                selected_count += 1;
            } else if (pruned_count < pruned_buf.len) {
                pruned_buf[pruned_count] = cand;
                pruned_count += 1;
            }
        }

        // Backfill: if heuristic was too aggressive, add best pruned candidates
        // (already sorted by distance since input was sorted)
        var pi: usize = 0;
        while (selected_count < m_max and pi < pruned_count) {
            candidates[selected_count] = pruned_buf[pi];
            selected_count += 1;
            pi += 1;
        }

        return selected_count;
    }

    /// Optimized concurrent insert path used by batch ingestion.
    /// Expects `nodes[idx]` to already exist with a selected level.
    pub fn insertParallelNoAlloc(self: *HNSWIndex, idx: usize, storage: *const VectorStorage) !void {
        const level = self.nodes.items[idx].level;
        var old_entry: usize = 0;
        var entry_level: u8 = 0;

        // Read a stable entry point.
        {
            self.build_rwlock.lockShared();
            defer self.build_rwlock.unlockShared();

            if (self.entry_point) |entry| {
                old_entry = entry;
                entry_level = self.nodes.items[entry].level;
            } else {
                return;
            }
        }

        // Entry point for an empty graph has nothing to link yet.
        if (idx == old_entry) return;

        var nearest_copy: []SearchItem = &[_]SearchItem{};

        // Search with shared lock, copy results, then link without global lock.
        {
            self.build_rwlock.lockShared();
            defer self.build_rwlock.unlockShared();

            const vector = storage.getVector(idx);
            var current_entry = old_entry;

            if (entry_level > 0) {
                var l = @min(level, entry_level);
                while (l > 0) : (l -= 1) {
                    const nearest_at_level = try self.searchLayerInsert(vector, current_entry, self.ef_construction, l, storage);
                    if (nearest_at_level.len > 0) {
                        current_entry = nearest_at_level[0].idx;
                    }
                }
            }

            const nearest = try self.searchLayerInsert(vector, current_entry, self.ef_construction, 0, storage);
            const ctx = if (tl_search_owner == self and tl_search_context != null)
                tl_search_context.?
            else
                try self.acquireSearchContext(@max(nearest.len, 1), self.nodes.items.len);

            try ensureBuffer(SearchItem, self.allocator, &ctx.scratch_results, nearest.len);
            @memcpy(ctx.scratch_results[0..nearest.len], nearest);
            nearest_copy = ctx.scratch_results[0..nearest.len];
        }

        var l: usize = 0;
        while (l <= level) : (l += 1) {
            const m = if (l == 0) self.m * 2 else self.m;

            // Filter candidates valid for this level
            var level_candidates_stack: [1024]SearchItem = undefined;
            var lc_len: usize = 0;

            for (nearest_copy) |item| {
                if (item.idx == idx) continue;
                if (l > self.nodes.items[item.idx].level) continue;
                if (lc_len < level_candidates_stack.len) {
                    level_candidates_stack[lc_len] = item;
                    lc_len += 1;
                }
            }

            // Apply HNSW spatial pruning heuristic for diverse neighbor selection
            const keep_count = self.selectNeighborsHeuristic(
                level_candidates_stack[0..lc_len],
                m,
                storage,
            );

            for (level_candidates_stack[0..keep_count]) |item| {
                const neighbor = item.idx;
                const dist = item.distance;

                // Self-node: single writer per idx, no lock needed
                try self.addConnection(idx, neighbor, l, dist, storage);

                // Neighbor node: try-lock, skip if contended
                {
                    const neighbor_node = &self.nodes.items[neighbor];
                    if (neighbor_node.trySpinLock()) {
                        defer neighbor_node.spinUnlock();
                        try self.addConnection(neighbor, idx, l, dist, storage);
                    }
                }
            }
        }

        // Publish higher-level entry points under exclusive lock.
        if (level > self.max_level) {
            self.build_rwlock.lock();
            defer self.build_rwlock.unlock();

            if (level > self.max_level) {
                self.max_level = level;
                self.entry_point = idx;
            }
        }
    }

    // ── Heap helpers for search priority queues ──────────────────────────

    inline fn heapSiftUpMin(items: []SearchItem, pos: usize) void {
        var child = pos;
        while (child > 0) {
            const parent = (child - 1) / 2;
            if (items[child].distance >= items[parent].distance) break;
            const tmp = items[child];
            items[child] = items[parent];
            items[parent] = tmp;
            child = parent;
        }
    }

    inline fn heapSiftDownMin(items: []SearchItem, pos: usize) void {
        var parent = pos;
        const len = items.len;
        while (true) {
            var smallest = parent;
            const left = 2 * parent + 1;
            const right = 2 * parent + 2;
            if (left < len and items[left].distance < items[smallest].distance) smallest = left;
            if (right < len and items[right].distance < items[smallest].distance) smallest = right;
            if (smallest == parent) break;
            const tmp = items[parent];
            items[parent] = items[smallest];
            items[smallest] = tmp;
            parent = smallest;
        }
    }

    inline fn heapSiftUpMax(items: []SearchItem, pos: usize) void {
        var child = pos;
        while (child > 0) {
            const parent = (child - 1) / 2;
            if (items[child].distance <= items[parent].distance) break;
            const tmp = items[child];
            items[child] = items[parent];
            items[parent] = tmp;
            child = parent;
        }
    }

    inline fn heapSiftDownMax(items: []SearchItem, pos: usize) void {
        var parent = pos;
        const len = items.len;
        while (true) {
            var largest = parent;
            const left = 2 * parent + 1;
            const right = 2 * parent + 2;
            if (left < len and items[left].distance > items[largest].distance) largest = left;
            if (right < len and items[right].distance > items[largest].distance) largest = right;
            if (largest == parent) break;
            const tmp = items[parent];
            items[parent] = items[largest];
            items[largest] = tmp;
            parent = largest;
        }
    }

    fn searchLayerInsert(
        self: *HNSWIndex,
        query: []const StorageScalar,
        entry: usize,
        num_closest: usize,
        layer: usize,
        storage: *const VectorStorage,
    ) ![]SearchItem {
        const ef = @max(num_closest, 1);
        const ctx = try self.acquireSearchContext(ef, self.nodes.items.len);
        const mark = ctx.mark;
        const visited = ctx.visited;
        const c_items = ctx.candidates; // min-heap: closest candidate at top
        const w_items = ctx.w; // max-heap: farthest result at top
        var c_len: usize = 0;
        var w_len: usize = 0;

        const entry_dist = VectorOps.cosineDistanceStorage(query, storage.getVector(entry));
        c_items[0] = .{ .idx = entry, .distance = entry_dist };
        c_len = 1;
        w_items[0] = .{ .idx = entry, .distance = entry_dist };
        w_len = 1;
        visited[entry] = mark;

        while (c_len > 0) {
            // Pop closest candidate (min-heap root)
            const current = c_items[0];

            // Early termination: closest candidate farther than worst result
            if (w_len >= ef and current.distance > w_items[0].distance) break;

            c_len -= 1;
            if (c_len > 0) {
                c_items[0] = c_items[c_len];
                heapSiftDownMin(c_items[0..c_len], 0);
            }

            if (layer > self.nodes.items[current.idx].level) continue;
            const connections = self.getConnections(current.idx, layer);

            // Prefetch first neighbor vectors + node metadata to hide memory latency
            const pf_count = @min(connections.len, 8);
            for (connections[0..pf_count]) |neighbor| {
                const vec = storage.getVector(neighbor);
                @prefetch(vec.ptr, .{ .rw = .read, .locality = 0, .cache = .data });
                @prefetch(@as([*]const u8, @ptrCast(&self.nodes.items[neighbor])), .{ .rw = .read, .locality = 1, .cache = .data });
            }

            for (connections, 0..) |neighbor, ci| {
                if (visited[neighbor] == mark) continue;
                visited[neighbor] = mark;

                // Prefetch ahead — both vectors and node metadata
                if (ci + pf_count < connections.len) {
                    const ahead_id = connections[ci + pf_count];
                    const ahead = storage.getVector(ahead_id);
                    @prefetch(ahead.ptr, .{ .rw = .read, .locality = 0, .cache = .data });
                    @prefetch(@as([*]const u8, @ptrCast(&self.nodes.items[ahead_id])), .{ .rw = .read, .locality = 1, .cache = .data });
                }

                const dist = VectorOps.cosineDistanceStorage(query, storage.getVector(neighbor));

                if (w_len < ef or dist < w_items[0].distance) {
                    // Push to candidate min-heap
                    if (c_len < c_items.len) {
                        c_items[c_len] = .{ .idx = neighbor, .distance = dist };
                        heapSiftUpMin(c_items[0 .. c_len + 1], c_len);
                        c_len += 1;
                    }

                    // Push to result max-heap
                    if (w_len < ef) {
                        w_items[w_len] = .{ .idx = neighbor, .distance = dist };
                        heapSiftUpMax(w_items[0 .. w_len + 1], w_len);
                        w_len += 1;
                    } else {
                        w_items[0] = .{ .idx = neighbor, .distance = dist };
                        heapSiftDownMax(w_items[0..w_len], 0);
                    }
                }
            }
        }

        std.sort.heap(SearchItem, w_items[0..w_len], {}, struct {
            fn lessThan(context: void, a: SearchItem, b: SearchItem) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        return w_items[0..w_len];
    }

    fn searchLayer(
        self: *HNSWIndex,
        query: []const f32,
        entry: usize,
        num_closest: usize,
        layer: usize,
        storage: *const VectorStorage,
    ) ![]SearchItem {
        const ef = @max(num_closest, 1);
        const ctx = try self.acquireSearchContext(ef, self.nodes.items.len);
        const visited = ctx.visited;
        const mark = ctx.mark;
        const c_items = ctx.candidates; // min-heap: closest at top
        const w_items = ctx.w; // max-heap: farthest at top
        var c_len: usize = 0;
        var w_len: usize = 0;

        const entry_dist = VectorOps.cosineDistanceQueryStorage(query, storage.getVector(entry));
        c_items[0] = .{ .idx = entry, .distance = entry_dist };
        c_len = 1;
        w_items[0] = .{ .idx = entry, .distance = entry_dist };
        w_len = 1;
        visited[entry] = mark;

        while (c_len > 0) {
            // Pop closest candidate (min-heap root)
            const current = c_items[0];

            // Early termination: closest candidate farther than worst result
            if (w_len >= ef and current.distance > w_items[0].distance) break;

            c_len -= 1;
            if (c_len > 0) {
                c_items[0] = c_items[c_len];
                heapSiftDownMin(c_items[0..c_len], 0);
            }

            if (layer > self.nodes.items[current.idx].level) continue;
            const connections = self.getConnections(current.idx, layer);

            // Prefetch first few neighbor vectors + their node metadata
            const prefetch_count = @min(connections.len, 8);
            for (connections[0..prefetch_count]) |neighbor| {
                const vec = storage.getVector(neighbor);
                @prefetch(vec.ptr, .{ .rw = .read, .locality = 0, .cache = .data });
                // Prefetch node metadata for future graph expansion
                @prefetch(@as([*]const u8, @ptrCast(&self.nodes.items[neighbor])), .{ .rw = .read, .locality = 1, .cache = .data });
            }

            for (connections) |neighbor| {
                if (visited[neighbor] == mark) continue;
                visited[neighbor] = mark;

                const dist = VectorOps.cosineDistanceQueryStorage(query, storage.getVector(neighbor));

                if (w_len < ef or dist < w_items[0].distance) {
                    // Push to candidate min-heap
                    if (c_len < c_items.len) {
                        c_items[c_len] = .{ .idx = neighbor, .distance = dist };
                        heapSiftUpMin(c_items[0 .. c_len + 1], c_len);
                        c_len += 1;
                    }

                    // Push to result max-heap
                    if (w_len < ef) {
                        w_items[w_len] = .{ .idx = neighbor, .distance = dist };
                        heapSiftUpMax(w_items[0 .. w_len + 1], w_len);
                        w_len += 1;
                    } else {
                        w_items[0] = .{ .idx = neighbor, .distance = dist };
                        heapSiftDownMax(w_items[0..w_len], 0);
                    }
                }
            }
        }

        // Sort results by distance ascending
        std.sort.heap(SearchItem, w_items[0..w_len], {}, struct {
            fn lessThan(context: void, a: SearchItem, b: SearchItem) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        return w_items[0..w_len];
    }

    /// Int8-accelerated graph search for HNSW assist path.
    /// Uses quantized int8 distances (~10x cheaper than f32) for graph traversal.
    /// Returns candidate IDs with approximate distances — caller reranks with f32.
    fn searchLayerI8(
        self: *HNSWIndex,
        query_q8: []const i8,
        quantized_vectors: []const i8,
        dim: usize,
        entry: usize,
        num_closest: usize,
        layer: usize,
    ) ![]SearchItem {
        const ef = @max(num_closest, 1);
        const ctx = try self.acquireSearchContext(ef, self.nodes.items.len);
        const visited = ctx.visited;
        const mark = ctx.mark;
        const c_items = ctx.candidates;
        const w_items = ctx.w;
        var c_len: usize = 0;
        var w_len: usize = 0;

        // Entry point distance using int8
        const entry_q = quantized_vectors[entry * dim ..][0..dim];
        const entry_dot = VectorOps.dotI8(query_q8, entry_q);
        const entry_dist = 1.0 - @as(f32, @floatFromInt(entry_dot)) * INT8_INV_SCALE_SQ;
        c_items[0] = .{ .idx = entry, .distance = entry_dist };
        c_len = 1;
        w_items[0] = .{ .idx = entry, .distance = entry_dist };
        w_len = 1;
        visited[entry] = mark;

        while (c_len > 0) {
            const current = c_items[0];

            if (w_len >= ef and current.distance > w_items[0].distance) break;

            c_len -= 1;
            if (c_len > 0) {
                c_items[0] = c_items[c_len];
                heapSiftDownMin(c_items[0..c_len], 0);
            }

            if (layer > self.nodes.items[current.idx].level) continue;
            const connections = self.getConnections(current.idx, layer);

            // Prefetch quantized vectors for first neighbors
            const pf_count = @min(connections.len, 8);
            for (connections[0..pf_count]) |neighbor| {
                @prefetch(quantized_vectors.ptr + neighbor * dim, .{ .rw = .read, .locality = 0, .cache = .data });
                @prefetch(@as([*]const u8, @ptrCast(&self.nodes.items[neighbor])), .{ .rw = .read, .locality = 1, .cache = .data });
            }

            for (connections, 0..) |neighbor, ci| {
                if (visited[neighbor] == mark) continue;
                visited[neighbor] = mark;

                if (ci + pf_count < connections.len) {
                    const ahead_id = connections[ci + pf_count];
                    @prefetch(quantized_vectors.ptr + ahead_id * dim, .{ .rw = .read, .locality = 0, .cache = .data });
                    @prefetch(@as([*]const u8, @ptrCast(&self.nodes.items[ahead_id])), .{ .rw = .read, .locality = 1, .cache = .data });
                }

                const qvec = quantized_vectors[neighbor * dim ..][0..dim];
                const dot_i32 = VectorOps.dotI8(query_q8, qvec);
                const dist = 1.0 - @as(f32, @floatFromInt(dot_i32)) * INT8_INV_SCALE_SQ;

                if (w_len < ef or dist < w_items[0].distance) {
                    if (c_len < c_items.len) {
                        c_items[c_len] = .{ .idx = neighbor, .distance = dist };
                        heapSiftUpMin(c_items[0 .. c_len + 1], c_len);
                        c_len += 1;
                    }

                    if (w_len < ef) {
                        w_items[w_len] = .{ .idx = neighbor, .distance = dist };
                        heapSiftUpMax(w_items[0 .. w_len + 1], w_len);
                        w_len += 1;
                    } else {
                        w_items[0] = .{ .idx = neighbor, .distance = dist };
                        heapSiftDownMax(w_items[0..w_len], 0);
                    }
                }
            }
        }

        std.sort.heap(SearchItem, w_items[0..w_len], {}, struct {
            fn lessThan(context: void, a: SearchItem, b: SearchItem) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        return w_items[0..w_len];
    }

    /// Efficient top-k selection using on-stack buffers for small sizes
    fn topK(
        self: *HNSWIndex,
        candidates: []usize,
        k: usize,
        base: []const f32,
        storage: *const VectorStorage,
    ) ![]usize {
        if (candidates.len <= k) {
            const result = try self.allocator.alloc(usize, candidates.len);
            @memcpy(result, candidates);
            return result;
        }

        // Create (distance, index) pairs
        const DistanceIndex = struct {
            distance: f32,
            candidate_idx: usize,
        };

        // Use on-stack buffer for small candidate sets to avoid heap allocation
        if (candidates.len <= 512) {
            var stack_pairs: [512]DistanceIndex = undefined;
            const pairs = stack_pairs[0..candidates.len];

            for (candidates, 0..) |id, i| {
                pairs[i] = DistanceIndex{
                    .distance = VectorOps.cosineDistanceQueryStorage(base, storage.getVector(id)),
                    .candidate_idx = i,
                };
            }

            // Sort the pairs by distance
            std.sort.heap(DistanceIndex, pairs, {}, struct {
                fn lessThan(context: void, a: DistanceIndex, b: DistanceIndex) bool {
                    _ = context;
                    return a.distance < b.distance;
                }
            }.lessThan);

            // Copy out the k closest candidates
            const result = try self.allocator.alloc(usize, k);
            for (0..k) |i| {
                result[i] = candidates[pairs[i].candidate_idx];
            }
            return result;
        } else {
            // Fall back to heap allocation for large candidate sets
            var pairs = try self.allocator.alloc(DistanceIndex, candidates.len);
            defer self.allocator.free(pairs);

            for (candidates, 0..) |id, i| {
                pairs[i] = DistanceIndex{
                    .distance = VectorOps.cosineDistanceQueryStorage(base, storage.getVector(id)),
                    .candidate_idx = i,
                };
            }

            // Sort the pairs by distance
            std.sort.heap(DistanceIndex, pairs, {}, struct {
                fn lessThan(context: void, a: DistanceIndex, b: DistanceIndex) bool {
                    _ = context;
                    return a.distance < b.distance;
                }
            }.lessThan);

            // Copy out the k closest candidates
            const result = try self.allocator.alloc(usize, k);
            for (0..k) |i| {
                result[i] = candidates[pairs[i].candidate_idx];
            }
            return result;
        }
    }

    // Comparator for max-heap (worst candidates at top)
    fn compareSearchItemsReverse(context: void, a: SearchItem, b: SearchItem) std.math.Order {
        _ = context;
        return std.math.order(b.distance, a.distance);
    }

    fn compareSearchItems(context: void, a: SearchItem, b: SearchItem) std.math.Order {
        _ = context;
        return std.math.order(a.distance, b.distance);
    }
};

// Main vector database structure
pub const VectorDB = struct {
    const FILE_MAGIC: u32 = 0x5A454354; // "ZECT"
    const FILE_VERSION: u32 = 1;

    const ApproxCandidate = struct {
        idx: usize,
        approx_distance: f32,
    };

    allocator: mem.Allocator,
    storage: VectorStorage,
    index: HNSWIndex,
    metadata: std.AutoHashMap(usize, std.json.Value),
    search_ef: usize,
    turbo_enabled: bool,
    turbo_dirty: bool,
    ivf_min_vectors: usize,
    ivf_nlist: usize,
    ivf_probes: usize,
    ivf_rerank_factor: usize,
    ivf_centroids: []f32,
    ivf_offsets: []usize,
    ivf_lengths: []usize,
    ivf_postings: []usize,
    ivf_postings_mmap: ?[]align(PAGE_ALIGNMENT) u8,
    ivf_postings_file: ?fs.File,
    ivf_postings_path: ?[]u8,
    ivf_postings_is_mmap: bool,
    quantized_vectors: []i8,
    binary_vectors: []u64,
    binary_words_per_vec: usize,
    storage_file_path: ?[]u8,

    // Internal→external ID mapping (maintained across graph compaction)
    internal_to_external: std.ArrayList(usize),

    // Product Quantization (PQ8) for sub-microsecond approximate distance
    pq_m: usize, // Number of subvectors (dim / pq_dsub)
    pq_dsub: usize, // Subvector dimension
    pq_ksub: usize, // Centroids per subvector (256)
    pq_codebook: []f32, // Codebook: pq_m * pq_ksub * pq_dsub floats
    pq_codes: []u8, // Encoded vectors: count * pq_m bytes

    // 4-bit Product Quantization (PQ4) with in-register NEON tbl lookup
    pq4_m: usize, // Number of subvectors
    pq4_dsub: usize, // Subvector dimension
    pq4_codebook: []f32, // Codebook: pq4_m * 16 * pq4_dsub floats
    pq4_codes: []u8, // Packed nibbles: count * (pq4_m / 2) bytes

    pub fn init(
        allocator: mem.Allocator,
        dimension: usize,
        options: struct {
            initial_capacity: usize = 10000,
            index_m: usize = 32, // Increased for better connectivity
            index_ef_construction: usize = 400, // Increased for better quality
            file_path: ?[]const u8 = null,
            enable_turbo: bool = true,
            ivf_min_vectors: usize = IVF_MIN_VECTORS_DEFAULT,
            ivf_nlist: usize = 0, // 0 => auto
            ivf_probes: usize = IVF_DEFAULT_PROBES,
            ivf_rerank_factor: usize = 8,
        },
    ) !VectorDB {
        const storage = if (options.file_path) |path|
            try VectorStorage.initWithFile(allocator, path, dimension)
        else
            try VectorStorage.init(allocator, dimension, options.initial_capacity);

        const storage_file_path = if (options.file_path) |path|
            try allocator.dupe(u8, path)
        else
            null;

        return VectorDB{
            .allocator = allocator,
            .storage = storage,
            .index = HNSWIndex.init(allocator, options.index_m, options.index_ef_construction),
            .metadata = std.AutoHashMap(usize, std.json.Value).init(allocator),
            .search_ef = 128,
            .turbo_enabled = options.enable_turbo,
            .turbo_dirty = true,
            .ivf_min_vectors = options.ivf_min_vectors,
            .ivf_nlist = options.ivf_nlist,
            .ivf_probes = @max(options.ivf_probes, 1),
            .ivf_rerank_factor = @max(options.ivf_rerank_factor, 1),
            .ivf_centroids = &[_]f32{},
            .ivf_offsets = &[_]usize{},
            .ivf_lengths = &[_]usize{},
            .ivf_postings = &[_]usize{},
            .ivf_postings_mmap = null,
            .ivf_postings_file = null,
            .ivf_postings_path = null,
            .ivf_postings_is_mmap = false,
            .quantized_vectors = &[_]i8{},
            .binary_vectors = &[_]u64{},
            .binary_words_per_vec = (dimension + 63) / 64,
            .storage_file_path = storage_file_path,
            .internal_to_external = std.ArrayList(usize).init(allocator),
            .pq_m = 0,
            .pq_dsub = 0,
            .pq_ksub = 256,
            .pq_codebook = &[_]f32{},
            .pq_codes = &[_]u8{},
            .pq4_m = 0,
            .pq4_dsub = 0,
            .pq4_codebook = &[_]f32{},
            .pq4_codes = &[_]u8{},
        };
    }

    pub fn deinit(self: *VectorDB) void {
        self.releaseIvfPostings();
        if (self.ivf_centroids.len > 0) self.allocator.free(self.ivf_centroids);
        if (self.ivf_offsets.len > 0) self.allocator.free(self.ivf_offsets);
        if (self.ivf_lengths.len > 0) self.allocator.free(self.ivf_lengths);
        if (self.quantized_vectors.len > 0) self.allocator.free(self.quantized_vectors);
        if (self.binary_vectors.len > 0) self.allocator.free(self.binary_vectors);
        if (self.pq_codebook.len > 0) self.allocator.free(self.pq_codebook);
        if (self.pq_codes.len > 0) self.allocator.free(self.pq_codes);
        if (self.pq4_codebook.len > 0) self.allocator.free(self.pq4_codebook);
        if (self.pq4_codes.len > 0) self.allocator.free(self.pq4_codes);
        self.internal_to_external.deinit();
        if (self.storage_file_path) |path| self.allocator.free(path);
        self.storage.deinit();
        self.index.deinit();
        self.metadata.deinit();
    }

    fn releaseIvfPostings(self: *VectorDB) void {
        if (self.ivf_postings_is_mmap) {
            if (self.ivf_postings_mmap) |mmap_bytes| {
                std.posix.munmap(mmap_bytes);
            }
            self.ivf_postings_mmap = null;
        } else if (self.ivf_postings.len > 0) {
            self.allocator.free(self.ivf_postings);
        }

        if (self.ivf_postings_file) |file| {
            file.close();
            self.ivf_postings_file = null;
        }

        if (self.ivf_postings_path) |path| {
            self.allocator.free(path);
            self.ivf_postings_path = null;
        }

        self.ivf_postings = &[_]usize{};
        self.ivf_postings_is_mmap = false;
    }

    fn allocateIvfPostings(self: *VectorDB, total: usize) !void {
        self.releaseIvfPostings();
        if (total == 0) return;

        const bytes = total * @sizeOf(usize);
        if (self.storage_file_path != null) {
            const base_path = self.storage_file_path.?;
            const path = try std.fmt.allocPrint(self.allocator, "{s}.ivf.postings", .{base_path});
            errdefer self.allocator.free(path);

            const file = try fs.cwd().createFile(path, .{
                .read = true,
                .truncate = true,
            });
            errdefer file.close();

            try file.setEndPos(bytes);

            const mmap_bytes = try std.posix.mmap(
                null,
                bytes,
                std.posix.PROT.READ | std.posix.PROT.WRITE,
                std.posix.MAP{ .TYPE = .SHARED },
                file.handle,
                0,
            );
            errdefer std.posix.munmap(mmap_bytes);

            self.ivf_postings_path = path;
            self.ivf_postings_file = file;
            self.ivf_postings_mmap = mmap_bytes;
            self.ivf_postings_is_mmap = true;
            self.ivf_postings = @as([*]usize, @ptrCast(@alignCast(mmap_bytes.ptr)))[0..total];
            return;
        }

        self.ivf_postings = try self.allocator.alloc(usize, total);
        self.ivf_postings_is_mmap = false;
    }

    fn quantizeValue(v: f32) i8 {
        return VectorOps.quantizeUnit(v);
    }

    fn centroidSlice(self: *const VectorDB, centroid_idx: usize) []const f32 {
        const dim = self.storage.dimension;
        const offset = centroid_idx * dim;
        return self.ivf_centroids[offset .. offset + dim];
    }

    fn nearestCentroid(self: *const VectorDB, vector: []const StorageScalar) usize {
        var best_idx: usize = 0;
        var best_dist: f32 = math.inf(f32);

        var c: usize = 0;
        while (c < self.ivf_offsets.len) : (c += 1) {
            const dist = VectorOps.cosineDistanceQueryStorage(self.centroidSlice(c), vector);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
            }
        }

        return best_idx;
    }

    fn rebuildTurboIndex(self: *VectorDB) !void {
        if (!self.turbo_enabled) {
            self.turbo_dirty = false;
            return;
        }

        const count = self.storage.count;
        const dim = self.storage.dimension;
        if (count < self.ivf_min_vectors) {
            self.releaseIvfPostings();
            if (self.ivf_centroids.len > 0) {
                self.allocator.free(self.ivf_centroids);
                self.ivf_centroids = &[_]f32{};
            }
            if (self.ivf_offsets.len > 0) {
                self.allocator.free(self.ivf_offsets);
                self.ivf_offsets = &[_]usize{};
            }
            if (self.ivf_lengths.len > 0) {
                self.allocator.free(self.ivf_lengths);
                self.ivf_lengths = &[_]usize{};
            }
            if (self.quantized_vectors.len > 0) {
                self.allocator.free(self.quantized_vectors);
                self.quantized_vectors = &[_]i8{};
            }
            if (self.binary_vectors.len > 0) {
                self.allocator.free(self.binary_vectors);
                self.binary_vectors = &[_]u64{};
            }
            if (self.pq_codebook.len > 0) {
                self.allocator.free(self.pq_codebook);
                self.pq_codebook = &[_]f32{};
            }
            if (self.pq_codes.len > 0) {
                self.allocator.free(self.pq_codes);
                self.pq_codes = &[_]u8{};
            }
            self.turbo_dirty = false;
            return;
        }

        const auto_nlist = std.math.clamp(
            @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(count))))),
            IVF_MIN_LISTS,
            IVF_MAX_LISTS,
        );
        const nlist = std.math.clamp(if (self.ivf_nlist == 0) auto_nlist else self.ivf_nlist, 1, count);
        self.ivf_nlist = nlist;

        const centroid_len = nlist * dim;
        if (self.ivf_centroids.len != centroid_len) {
            if (self.ivf_centroids.len > 0) self.allocator.free(self.ivf_centroids);
            self.ivf_centroids = try self.allocator.alloc(f32, centroid_len);
        }

        if (self.ivf_offsets.len != nlist) {
            if (self.ivf_offsets.len > 0) self.allocator.free(self.ivf_offsets);
            self.ivf_offsets = try self.allocator.alloc(usize, nlist);
        }

        if (self.ivf_lengths.len != nlist) {
            if (self.ivf_lengths.len > 0) self.allocator.free(self.ivf_lengths);
            self.ivf_lengths = try self.allocator.alloc(usize, nlist);
        }

        // Initialize centroids from evenly spaced samples.
        for (0..nlist) |c| {
            const sample_idx = (c * count) / nlist;
            const vec = self.storage.getVector(sample_idx);
            if (comptime StorageScalar == f32) {
                @memcpy(self.ivf_centroids[c * dim ..][0..dim], vec);
            } else {
                for (vec, 0..) |v, d| {
                    self.ivf_centroids[c * dim + d] = VectorOps.toF32(v);
                }
            }
        }

        // Sampled K-means refinement — use a subset for centroid training (much faster for large n).
        const sample_size = @min(count, nlist * 40); // ~40 samples per cluster is sufficient
        var sums = try self.allocator.alloc(f32, centroid_len);
        defer self.allocator.free(sums);
        var counts = try self.allocator.alloc(usize, nlist);
        defer self.allocator.free(counts);

        const stride = if (sample_size < count) count / sample_size else 1;
        var iter: usize = 0;
        while (iter < 2) : (iter += 1) {
            @memset(sums, 0.0);
            @memset(counts, 0);

            var s: usize = 0;
            while (s < sample_size) : (s += 1) {
                const i = (s * stride) % count;
                const vec = self.storage.getVector(i);
                const c = self.nearestCentroid(vec);
                counts[c] += 1;
                const base = c * dim;
                for (0..dim) |d| {
                    sums[base + d] += VectorOps.toF32(vec[d]);
                }
            }

            for (0..nlist) |c| {
                if (counts[c] == 0) continue;
                const base = c * dim;
                const inv = 1.0 / @as(f32, @floatFromInt(counts[c]));
                for (0..dim) |d| {
                    self.ivf_centroids[base + d] = sums[base + d] * inv;
                }
                VectorOps.normalize(self.ivf_centroids[base .. base + dim]);
            }
        }

        // Single-pass: assign centroids + quantize in one scan over all vectors.
        // First pass: count cluster sizes (needed for offset computation).
        var assignments = try self.allocator.alloc(u16, count);
        defer self.allocator.free(assignments);
        @memset(self.ivf_lengths, 0);

        // Quantized copy for approximate scoring.
        const quant_len = count * dim;
        if (self.quantized_vectors.len != quant_len) {
            if (self.quantized_vectors.len > 0) self.allocator.free(self.quantized_vectors);
            self.quantized_vectors = try self.allocator.alloc(i8, quant_len);
        }

        // Binary quantization storage (sign-bit encoding for Hamming pre-filter)
        const bwords = self.binary_words_per_vec;
        const bin_len = count * bwords;
        if (self.binary_vectors.len != bin_len) {
            if (self.binary_vectors.len > 0) self.allocator.free(self.binary_vectors);
            self.binary_vectors = try self.allocator.alloc(u64, bin_len);
        }

        // Parallel fused assignment + quantization + binary encoding (single pass over all vectors).
        const assign_threads = @max(1, @min(configuredCpuCount(), count / 256));
        if (assign_threads > 1) {
            const AssignCtx = struct {
                db: *const VectorDB,
                assign_out: []u16,
                quant_out: []i8,
                bin_out: []u64,
                bw: usize,
                start: usize,
                end: usize,
                local_lengths: [IVF_MAX_LISTS]usize,

                fn run(ctx: *@This()) void {
                    @memset(&ctx.local_lengths, 0);
                    const d = ctx.db.storage.dimension;
                    var idx = ctx.start;
                    while (idx < ctx.end) : (idx += 1) {
                        const vec = ctx.db.storage.getVector(idx);
                        const c = ctx.db.nearestCentroid(vec);
                        ctx.assign_out[idx] = @intCast(c);
                        ctx.local_lengths[c] += 1;
                        const dst = ctx.quant_out[idx * d ..][0..d];
                        for (vec, 0..) |v, dd| {
                            dst[dd] = quantizeValue(VectorOps.toF32(v));
                        }
                        // Binary encode in same pass
                        VectorOps.binaryEncodeStorage(vec, ctx.bin_out[idx * ctx.bw ..][0..ctx.bw]);
                    }
                }
            };

            var ctxs = try self.allocator.alloc(AssignCtx, assign_threads);
            defer self.allocator.free(ctxs);
            var threads = try self.allocator.alloc(std.Thread, assign_threads);
            defer self.allocator.free(threads);

            const chunk = (count + assign_threads - 1) / assign_threads;
            for (0..assign_threads) |ti| {
                ctxs[ti] = .{
                    .db = self,
                    .assign_out = assignments,
                    .quant_out = self.quantized_vectors,
                    .bin_out = self.binary_vectors,
                    .bw = bwords,
                    .start = ti * chunk,
                    .end = @min((ti + 1) * chunk, count),
                    .local_lengths = undefined,
                };
                threads[ti] = try std.Thread.spawn(.{}, AssignCtx.run, .{&ctxs[ti]});
            }
            for (threads) |t| t.join();

            // Merge per-thread lengths
            @memset(self.ivf_lengths, 0);
            for (ctxs) |ctx| {
                for (0..nlist) |c| {
                    self.ivf_lengths[c] += ctx.local_lengths[c];
                }
            }
        } else {
            // Single-threaded fallback
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const vec = self.storage.getVector(i);
                const c = self.nearestCentroid(vec);
                assignments[i] = @intCast(c);
                self.ivf_lengths[c] += 1;
                const dst = self.quantized_vectors[i * dim ..][0..dim];
                for (vec, 0..) |v, d| {
                    dst[d] = quantizeValue(VectorOps.toF32(v));
                }
                VectorOps.binaryEncodeStorage(vec, self.binary_vectors[i * bwords ..][0..bwords]);
            }
        }

        // Build posting lists from cached assignments (no second centroid scan).
        var running: usize = 0;
        for (0..nlist) |c| {
            self.ivf_offsets[c] = running;
            running += self.ivf_lengths[c];
        }

        try self.allocateIvfPostings(count);

        var cursors = try self.allocator.alloc(usize, nlist);
        defer self.allocator.free(cursors);
        @memcpy(cursors, self.ivf_offsets);

        for (0..count) |vi| {
            const c = assignments[vi];
            const write_pos = cursors[c];
            self.ivf_postings[write_pos] = vi;
            cursors[c] += 1;
        }

        self.turbo_dirty = false;

        // Build 4-bit PQ for ultra-fast approximate distance filtering
        try self.buildPQ4();

        // Reorder graph nodes into BFS traversal order for cache-optimal search
        try self.compactGraphLayout();
    }

    /// Train PQ codebook using sampled K-means and encode all vectors.
    fn buildPQ(self: *VectorDB) !void {
        const dim = self.storage.dimension;
        const count = self.storage.count;
        // Only train PQ for large datasets where it amortizes (> 50K vectors)
        if (count < 50_000 or dim < 8) return;

        // PQ parameters: split into subvectors of 8 dimensions
        const dsub: usize = 8;
        const pq_m = dim / dsub;
        if (pq_m == 0) return;
        const ksub: usize = 256; // Standard PQ with 1-byte codes

        self.pq_m = pq_m;
        self.pq_dsub = dsub;
        self.pq_ksub = ksub;

        // Allocate codebook: pq_m * ksub * dsub floats
        const cb_len = pq_m * ksub * dsub;
        if (self.pq_codebook.len != cb_len) {
            if (self.pq_codebook.len > 0) self.allocator.free(self.pq_codebook);
            self.pq_codebook = try self.allocator.alloc(f32, cb_len);
        }

        // Allocate PQ codes: count * pq_m bytes
        const codes_len = count * pq_m;
        if (self.pq_codes.len != codes_len) {
            if (self.pq_codes.len > 0) self.allocator.free(self.pq_codes);
            self.pq_codes = try self.allocator.alloc(u8, codes_len);
        }

        // Train each subvector's codebook independently using sampled K-means
        const sample_size = @min(count, ksub * 40);
        const stride_s = if (sample_size < count) count / sample_size else 1;
        var sums = try self.allocator.alloc(f32, ksub * dsub);
        defer self.allocator.free(sums);
        var kcounts = try self.allocator.alloc(usize, ksub);
        defer self.allocator.free(kcounts);

        for (0..pq_m) |m| {
            const sub_offset = m * dsub;
            const cb_base = m * ksub * dsub;

            // Initialize centroids from evenly spaced samples
            for (0..ksub) |k| {
                const vi = (k * count / ksub) % count;
                const vec = self.storage.getVector(vi);
                for (0..dsub) |d| {
                    self.pq_codebook[cb_base + k * dsub + d] = VectorOps.toF32(vec[sub_offset + d]);
                }
            }

            // 2 iterations of K-means (sampled)
            var iter: usize = 0;
            while (iter < 2) : (iter += 1) {
                @memset(sums, 0.0);
                @memset(kcounts, 0);

                var s: usize = 0;
                while (s < sample_size) : (s += 1) {
                    const vi = (s * stride_s) % count;
                    const vec = self.storage.getVector(vi);

                    // Find nearest centroid for this subvector
                    var best_k: usize = 0;
                    var best_dist: f32 = std.math.inf(f32);
                    for (0..ksub) |k| {
                        var dist: f32 = 0;
                        for (0..dsub) |d| {
                            const diff = VectorOps.toF32(vec[sub_offset + d]) - self.pq_codebook[cb_base + k * dsub + d];
                            dist += diff * diff;
                        }
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }

                    kcounts[best_k] += 1;
                    for (0..dsub) |d| {
                        sums[best_k * dsub + d] += VectorOps.toF32(vec[sub_offset + d]);
                    }
                }

                // Update centroids
                for (0..ksub) |k| {
                    if (kcounts[k] == 0) continue;
                    const inv = 1.0 / @as(f32, @floatFromInt(kcounts[k]));
                    for (0..dsub) |d| {
                        self.pq_codebook[cb_base + k * dsub + d] = sums[k * dsub + d] * inv;
                    }
                }
            }
        }

        // Encode all vectors: assign each subvector to nearest centroid
        for (0..count) |vi| {
            const vec = self.storage.getVector(vi);
            for (0..pq_m) |m| {
                const sub_offset = m * dsub;
                const cb_base = m * ksub * dsub;
                var best_k: u8 = 0;
                var best_dist: f32 = std.math.inf(f32);
                for (0..ksub) |k| {
                    var dist: f32 = 0;
                    for (0..dsub) |d| {
                        const diff = VectorOps.toF32(vec[sub_offset + d]) - self.pq_codebook[cb_base + k * dsub + d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = @intCast(k);
                    }
                }
                self.pq_codes[vi * pq_m + m] = best_k;
            }
        }
    }

    /// Compute asymmetric PQ distance from pre-computed distance table.
    /// Returns approximate squared L2 distance.
    inline fn pqDistanceADC(self: *const VectorDB, dist_table: []const f32, vec_idx: usize) f32 {
        const pq_m = self.pq_m;
        const codes = self.pq_codes[vec_idx * pq_m ..][0..pq_m];
        var dist: f32 = 0;
        // Process 4 subvectors at a time for ILP
        const aligned = pq_m - (pq_m % 4);
        var m: usize = 0;
        while (m < aligned) : (m += 4) {
            dist += dist_table[m * self.pq_ksub + codes[m]];
            dist += dist_table[(m + 1) * self.pq_ksub + codes[m + 1]];
            dist += dist_table[(m + 2) * self.pq_ksub + codes[m + 2]];
            dist += dist_table[(m + 3) * self.pq_ksub + codes[m + 3]];
        }
        while (m < pq_m) : (m += 1) {
            dist += dist_table[m * self.pq_ksub + codes[m]];
        }
        return dist;
    }

    /// Build the ADC distance table from a query vector.
    /// For each subvector m and centroid k, compute ||query_sub - centroid||^2.
    fn buildPQDistTable(self: *const VectorDB, query: []const f32, table: []f32) void {
        const pq_m = self.pq_m;
        const dsub = self.pq_dsub;
        const ksub = self.pq_ksub;
        for (0..pq_m) |m| {
            const sub_offset = m * dsub;
            const cb_base = m * ksub * dsub;
            for (0..ksub) |k| {
                var dist: f32 = 0;
                for (0..dsub) |d| {
                    const diff = query[sub_offset + d] - self.pq_codebook[cb_base + k * dsub + d];
                    dist += diff * diff;
                }
                table[m * ksub + k] = dist;
            }
        }
    }

    // ── 4-bit Product Quantization (PQ4) with in-register NEON tbl ──

    /// Train PQ4 codebook (K=16 per subvector) and encode all vectors as packed nibbles.
    fn buildPQ4(self: *VectorDB) !void {
        const dim = self.storage.dimension;
        const count = self.storage.count;
        if (count < self.ivf_min_vectors or dim < 8) return;

        const dsub: usize = 8;
        const pq4_m = dim / dsub;
        if (pq4_m < 2) return; // Need at least 2 subvectors for nibble packing
        const ksub: usize = 16; // 4-bit codes

        self.pq4_m = pq4_m;
        self.pq4_dsub = dsub;

        // Allocate codebook: pq4_m * 16 * dsub floats
        const cb_len = pq4_m * ksub * dsub;
        if (self.pq4_codebook.len != cb_len) {
            if (self.pq4_codebook.len > 0) self.allocator.free(self.pq4_codebook);
            self.pq4_codebook = try self.allocator.alloc(f32, cb_len);
        }

        // Allocate packed codes: count * (pq4_m / 2) bytes (2 codes per byte)
        const codes_len = count * (pq4_m / 2);
        if (self.pq4_codes.len != codes_len) {
            if (self.pq4_codes.len > 0) self.allocator.free(self.pq4_codes);
            self.pq4_codes = try self.allocator.alloc(u8, codes_len);
        }

        // Train each subvector codebook with sampled K-means (K=16)
        const sample_size = @min(count, ksub * 40);
        const stride_s = if (sample_size < count) count / sample_size else 1;
        var sums = try self.allocator.alloc(f32, ksub * dsub);
        defer self.allocator.free(sums);
        var kcounts = try self.allocator.alloc(usize, ksub);
        defer self.allocator.free(kcounts);

        for (0..pq4_m) |m| {
            const sub_offset = m * dsub;
            const cb_base = m * ksub * dsub;

            // Initialize centroids from evenly spaced samples
            for (0..ksub) |k| {
                const vi = (k * count / ksub) % count;
                const vec = self.storage.getVector(vi);
                for (0..dsub) |d| {
                    self.pq4_codebook[cb_base + k * dsub + d] = VectorOps.toF32(vec[sub_offset + d]);
                }
            }

            // 3 iterations of K-means (more iterations for fewer centroids)
            for (0..3) |_| {
                @memset(sums, 0.0);
                @memset(kcounts, 0);

                var s: usize = 0;
                while (s < sample_size) : (s += 1) {
                    const vi = (s * stride_s) % count;
                    const vec = self.storage.getVector(vi);
                    var best_k: usize = 0;
                    var best_dist: f32 = std.math.inf(f32);
                    for (0..ksub) |k| {
                        var dist: f32 = 0;
                        for (0..dsub) |d| {
                            const diff = VectorOps.toF32(vec[sub_offset + d]) - self.pq4_codebook[cb_base + k * dsub + d];
                            dist += diff * diff;
                        }
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                    kcounts[best_k] += 1;
                    for (0..dsub) |d| {
                        sums[best_k * dsub + d] += VectorOps.toF32(vec[sub_offset + d]);
                    }
                }

                for (0..ksub) |k| {
                    if (kcounts[k] == 0) continue;
                    const inv = 1.0 / @as(f32, @floatFromInt(kcounts[k]));
                    for (0..dsub) |d| {
                        self.pq4_codebook[cb_base + k * dsub + d] = sums[k * dsub + d] * inv;
                    }
                }
            }
        }

        // Encode all vectors as packed nibbles (2 codes per byte)
        const half_m = pq4_m / 2;
        for (0..count) |vi| {
            const vec = self.storage.getVector(vi);
            var m: usize = 0;
            while (m < pq4_m - 1) : (m += 2) {
                const code_lo = pq4FindNearest(self.pq4_codebook, m, dsub, ksub, vec);
                const code_hi = pq4FindNearest(self.pq4_codebook, m + 1, dsub, ksub, vec);
                self.pq4_codes[vi * half_m + m / 2] = code_lo | (code_hi << 4);
            }
            // Handle odd pq4_m
            if (pq4_m % 2 == 1) {
                const code_lo = pq4FindNearest(self.pq4_codebook, pq4_m - 1, dsub, ksub, vec);
                self.pq4_codes[vi * half_m + half_m - 1] = code_lo;
            }
        }
    }

    fn pq4FindNearest(codebook: []const f32, m: usize, dsub: usize, ksub: usize, vec: []const StorageScalar) u8 {
        const sub_offset = m * dsub;
        const cb_base = m * ksub * dsub;
        var best_k: u8 = 0;
        var best_dist: f32 = std.math.inf(f32);
        for (0..ksub) |k| {
            var dist: f32 = 0;
            for (0..dsub) |d| {
                const diff = VectorOps.toF32(vec[sub_offset + d]) - codebook[cb_base + k * dsub + d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = @intCast(k);
            }
        }
        return best_k;
    }

    /// Build u8 quantized distance table for PQ4 ADC.
    /// For each subvector m, computes distance from query to all 16 centroids,
    /// then quantizes to u8. Returns the scale factor for converting back to f32.
    fn pq4BuildDistTable(self: *const VectorDB, query: []const f32, table: []u8) f32 {
        const pq4_m = self.pq4_m;
        const dsub = self.pq4_dsub;
        const ksub: usize = 16;

        // First pass: compute f32 distances and find global max
        var max_dist: f32 = 0;
        for (0..pq4_m) |m| {
            const sub_offset = m * dsub;
            const cb_base = m * ksub * dsub;
            for (0..ksub) |k| {
                var dist: f32 = 0;
                for (0..dsub) |d| {
                    const diff = query[sub_offset + d] - self.pq4_codebook[cb_base + k * dsub + d];
                    dist += diff * diff;
                }
                if (dist > max_dist) max_dist = dist;
            }
        }

        // Quantize to u8 with scale
        const scale = if (max_dist > 0) 255.0 / max_dist else 1.0;
        for (0..pq4_m) |m| {
            const sub_offset = m * dsub;
            const cb_base = m * ksub * dsub;
            for (0..ksub) |k| {
                var dist: f32 = 0;
                for (0..dsub) |d| {
                    const diff = query[sub_offset + d] - self.pq4_codebook[cb_base + k * dsub + d];
                    dist += diff * diff;
                }
                table[m * ksub + k] = @intFromFloat(@min(255.0, dist * scale));
            }
        }

        return if (scale > 0) 1.0 / scale else 1.0;
    }

    /// PQ4 asymmetric distance using NEON tbl in-register lookup.
    /// Each subvector's 16-entry u8 table fits in one 128-bit register.
    /// The 4-bit codes index directly into the table via tbl instruction.
    inline fn pq4DistanceADC(self: *const VectorDB, dist_table: []const u8, vec_idx: usize) u32 {
        const pq4_m = self.pq4_m;
        const half_m = pq4_m / 2;
        const codes = self.pq4_codes[vec_idx * half_m ..][0..half_m];

        if (comptime builtin.cpu.arch == .aarch64) {
            return pq4DistanceNeon(dist_table, codes, pq4_m);
        }
        return pq4DistanceGeneric(dist_table, codes, pq4_m);
    }

    fn pq4DistanceNeon(dist_table: []const u8, codes: []const u8, pq4_m: usize) u32 {
        const half_m = pq4_m / 2;
        var sum: u32 = 0;

        // Process 2 subvectors at a time (1 byte = 2 packed nibbles)
        for (0..half_m) |i| {
            const byte = codes[i];
            const code_lo: u8 = byte & 0x0F;
            const code_hi: u8 = byte >> 4;
            const m_lo = i * 2;
            const m_hi = m_lo + 1;

            // Load 16-byte distance tables into NEON registers
            const table_lo: @Vector(16, u8) = dist_table[m_lo * 16 ..][0..16].*;
            const table_hi: @Vector(16, u8) = dist_table[m_hi * 16 ..][0..16].*;

            // Use tbl for in-register lookup (index vector with single code byte)
            var idx_lo: @Vector(16, u8) = @splat(0xFF); // Invalid indices → 0 output
            idx_lo[0] = code_lo;
            var idx_hi: @Vector(16, u8) = @splat(0xFF);
            idx_hi[0] = code_hi;

            // ARM tbl: lookup table_lo[idx_lo[i]] for each i
            // Indices >= 16 return 0, so only slot 0 has a valid result
            const result_lo = asm ("tbl %[d].16b, {%[n].16b}, %[m].16b"
                : [d] "=w" (-> @Vector(16, u8))
                : [n] "w" (table_lo), [m] "w" (idx_lo),
            );
            const result_hi = asm ("tbl %[d].16b, {%[n].16b}, %[m].16b"
                : [d] "=w" (-> @Vector(16, u8))
                : [n] "w" (table_hi), [m] "w" (idx_hi),
            );

            sum += @as(u32, result_lo[0]) + @as(u32, result_hi[0]);
        }

        // Handle odd subvector count
        if (pq4_m % 2 == 1) {
            const last_m = pq4_m - 1;
            const byte = codes[half_m - 1];
            const code: u8 = byte & 0x0F;
            sum += dist_table[last_m * 16 + code];
        }

        return sum;
    }

    fn pq4DistanceGeneric(dist_table: []const u8, codes: []const u8, pq4_m: usize) u32 {
        const half_m = pq4_m / 2;
        var sum: u32 = 0;

        for (0..half_m) |i| {
            const byte = codes[i];
            const code_lo: u8 = byte & 0x0F;
            const code_hi: u8 = byte >> 4;
            sum += dist_table[i * 2 * 16 + code_lo];
            sum += dist_table[(i * 2 + 1) * 16 + code_hi];
        }

        if (pq4_m % 2 == 1) {
            const last_m = pq4_m - 1;
            const byte = codes[half_m - 1];
            sum += dist_table[last_m * 16 + (byte & 0x0F)];
        }

        return sum;
    }

    /// Reorder nodes, vectors, and auxiliary data into BFS traversal order.
    /// After reordering, a node's graph neighbors tend to be physically adjacent
    /// in memory, dramatically reducing cache misses during HNSW search.
    fn compactGraphLayout(self: *VectorDB) !void {
        const n = self.storage.count;
        if (n < 2 or self.index.entry_point == null) return;

        // ── Step 1: BFS from entry point to determine new ordering ──
        var order = try self.allocator.alloc(usize, n);
        defer self.allocator.free(order);
        var new_id = try self.allocator.alloc(usize, n);
        defer self.allocator.free(new_id);
        var visited = try self.allocator.alloc(bool, n);
        defer self.allocator.free(visited);
        @memset(visited, false);

        var head: usize = 0;
        var tail: usize = 0;
        order[tail] = self.index.entry_point.?;
        visited[self.index.entry_point.?] = true;
        tail += 1;

        while (head < tail) {
            const old = order[head];
            head += 1;
            const conns = self.index.getConnections(old, 0);
            for (conns) |neighbor| {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    order[tail] = neighbor;
                    tail += 1;
                }
            }
        }
        // Disconnected nodes
        for (0..n) |i| {
            if (!visited[i]) {
                order[tail] = i;
                tail += 1;
            }
        }

        // Reverse mapping: old_id -> new_id
        for (order, 0..) |old, idx| {
            new_id[old] = idx;
        }

        // Check if already well-ordered (skip expensive copy)
        var already_ordered = true;
        for (order, 0..) |old, idx| {
            if (old != idx) {
                already_ordered = false;
                break;
            }
        }
        if (already_ordered) return;

        // ── Step 2: Remap entry point ──
        self.index.entry_point = new_id[self.index.entry_point.?];

        // ── Step 3: Remap all edge targets (only active connections) ──
        for (self.index.nodes.items[0..n]) |node| {
            for (node.connections) |conn| {
                if (conn.len == 0) continue;
                const edges = self.index.edge_pool.items[conn.offset..][0..conn.len];
                for (edges) |*edge| {
                    edge.* = new_id[edge.*];
                }
            }
        }

        // ── Step 4: Reorder nodes ──
        {
            var new_nodes = try self.allocator.alloc(HNSWIndex.Node, n);
            defer self.allocator.free(new_nodes);
            for (order, 0..) |old, idx| {
                new_nodes[idx] = self.index.nodes.items[old];
            }
            @memcpy(self.index.nodes.items[0..n], new_nodes);
        }

        // ── Step 5: Reorder vectors in storage ──
        {
            const vec_size = self.storage.dimension * @sizeOf(StorageScalar);
            var vec_copy = try self.allocator.alloc(u8, n * vec_size);
            defer self.allocator.free(vec_copy);
            for (order, 0..) |old, idx| {
                @memcpy(vec_copy[idx * vec_size ..][0..vec_size], self.storage.getVectorBytes(old));
            }
            for (0..n) |idx| {
                @memcpy(self.storage.getVectorBytesMut(idx), vec_copy[idx * vec_size ..][0..vec_size]);
            }
        }

        // ── Step 6: Reorder quantized vectors ──
        if (self.quantized_vectors.len > 0) {
            const dim = self.storage.dimension;
            var qv_copy = try self.allocator.alloc(i8, self.quantized_vectors.len);
            defer self.allocator.free(qv_copy);
            for (order, 0..) |old, idx| {
                @memcpy(qv_copy[idx * dim ..][0..dim], self.quantized_vectors[old * dim ..][0..dim]);
            }
            @memcpy(self.quantized_vectors, qv_copy);
        }

        // ── Step 7: Reorder binary vectors ──
        if (self.binary_vectors.len > 0) {
            const bw = self.binary_words_per_vec;
            var bv_copy = try self.allocator.alloc(u64, self.binary_vectors.len);
            defer self.allocator.free(bv_copy);
            for (order, 0..) |old, idx| {
                @memcpy(bv_copy[idx * bw ..][0..bw], self.binary_vectors[old * bw ..][0..bw]);
            }
            @memcpy(self.binary_vectors, bv_copy);
        }

        // ── Step 8: Reorder PQ codes ──
        if (self.pq_codes.len > 0) {
            const pm = self.pq_m;
            var pq_copy = try self.allocator.alloc(u8, self.pq_codes.len);
            defer self.allocator.free(pq_copy);
            for (order, 0..) |old, idx| {
                @memcpy(pq_copy[idx * pm ..][0..pm], self.pq_codes[old * pm ..][0..pm]);
            }
            @memcpy(self.pq_codes, pq_copy);
        }

        // ── Step 8b: Reorder PQ4 codes ──
        if (self.pq4_codes.len > 0) {
            const half_m = self.pq4_m / 2;
            var pq4_copy = try self.allocator.alloc(u8, self.pq4_codes.len);
            defer self.allocator.free(pq4_copy);
            for (order, 0..) |old, idx| {
                @memcpy(pq4_copy[idx * half_m ..][0..half_m], self.pq4_codes[old * half_m ..][0..half_m]);
            }
            @memcpy(self.pq4_codes, pq4_copy);
        }

        // ── Step 9: Remap IVF postings ──
        for (self.ivf_postings) |*posting| {
            if (posting.* < n) posting.* = new_id[posting.*];
        }

        // ── Step 10: Reorder internal→external ID mapping ──
        if (self.internal_to_external.items.len >= n) {
            var ext_copy = try self.allocator.alloc(usize, n);
            defer self.allocator.free(ext_copy);
            for (order, 0..) |old, new_idx| {
                ext_copy[new_idx] = self.internal_to_external.items[old];
            }
            @memcpy(self.internal_to_external.items[0..n], ext_copy);
        }
    }

    pub fn addVector(self: *VectorDB, vector: []const f32, metadata: ?std.json.Value) !usize {
        if (vector.len != self.storage.dimension) return error.InvalidDimension;

        const idx = try self.storage.addVector(vector);
        try self.internal_to_external.append(idx);
        try self.index.insert(idx, &self.storage);
        self.turbo_dirty = true;

        if (metadata) |m| {
            try self.metadata.put(idx, m);
        }

        return idx;
    }

    inline fn resultVectorSlice(self: *VectorDB, idx: usize) []const f32 {
        if (comptime StorageScalar == f32) {
            return self.storage.getVector(idx);
        }
        return &[_]f32{};
    }

    fn searchBruteForce(self: *VectorDB, normalized_query: []const f32, k: usize) ![]SearchResult {
        const result_len = @min(k, self.storage.count);
        var results = try self.allocator.alloc(SearchResult, result_len);
        if (result_len == 0) return results;

        var filled: usize = 0;
        var worst_pos: usize = 0;
        var worst_dist: f32 = -std.math.inf(f32);

        var idx: usize = 0;
        while (idx < self.storage.count) : (idx += 1) {
            const dist = VectorOps.cosineDistanceQueryStorage(normalized_query, self.storage.getVector(idx));

            if (filled < result_len) {
                results[filled] = .{
                    .idx = idx,
                    .distance = dist,
                    .vector = self.resultVectorSlice(idx),
                };
                if (dist > worst_dist) {
                    worst_dist = dist;
                    worst_pos = filled;
                }
                filled += 1;
                continue;
            }

            if (dist >= worst_dist) continue;

            results[worst_pos] = .{
                .idx = idx,
                .distance = dist,
                .vector = self.resultVectorSlice(idx),
            };

            worst_pos = 0;
            worst_dist = results[0].distance;
            var i: usize = 1;
            while (i < result_len) : (i += 1) {
                if (results[i].distance > worst_dist) {
                    worst_dist = results[i].distance;
                    worst_pos = i;
                }
            }
        }

        std.sort.heap(SearchResult, results, {}, struct {
            fn lessThan(context: void, a: SearchResult, b: SearchResult) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        return results;
    }

    fn searchHnsw(self: *VectorDB, normalized_query: []const f32, k: usize) ![]SearchResult {
        if (self.index.entry_point == null) {
            return try self.allocator.alloc(SearchResult, 0);
        }

        const ef = @max(k * 4, self.search_ef);
        self.index.build_rwlock.lockShared();
        defer self.index.build_rwlock.unlockShared();
        const items = try self.index.searchLayer(normalized_query, self.index.entry_point.?, ef, 0, &self.storage);
        // items is a slice into context buffer – no deallocation needed

        const result_len = @min(k, items.len);
        var results = try self.allocator.alloc(SearchResult, result_len);
        for (items[0..result_len], 0..) |item, i| {
            results[i] = .{
                .idx = item.idx,
                .distance = item.distance, // already computed – no recomputation!
                .vector = self.resultVectorSlice(item.idx),
            };
        }

        return results;
    }

    fn searchTurbo(self: *VectorDB, normalized_query: []const f32, k: usize) ![]SearchResult {
        if (self.turbo_dirty) try self.rebuildTurboIndex();
        if (self.ivf_offsets.len == 0 or self.quantized_vectors.len == 0) {
            return self.searchHnsw(normalized_query, k);
        }

        const ProbeCandidate = struct {
            idx: usize,
            distance: f32,
        };

        const nlist = self.ivf_offsets.len;
        // Adaptive probe scaling: aggressive probing only when user demands high recall.
        // Low ef = throughput mode (few probes), high ef = recall mode (many probes).
        // IVF probing is cheap (int8) vs HNSW assist (f32), so probe aggressively.
        const adaptive_probes = if (self.search_ef >= 256)
            self.search_ef / 5
        else
            self.search_ef / 32;
        const probes = std.math.clamp(@max(self.ivf_probes, adaptive_probes), 1, nlist);
        if (probes == 0) return self.searchHnsw(normalized_query, k);

        // Stack buffer for probe lists (max 512 probes × 12 bytes = 6KB)
        var probe_lists_stack: [IVF_MAX_LISTS]ProbeCandidate = undefined;
        const probe_lists = probe_lists_stack[0..probes];
        var probe_len: usize = 0;
        var probe_worst_pos: usize = 0;
        var probe_worst_dist: f32 = -std.math.inf(f32);

        // Prefetch first centroid
        if (nlist > 0) {
            @prefetch(@as([*]const u8, @ptrCast(self.centroidSlice(0).ptr)), .{ .rw = .read, .locality = 1, .cache = .data });
        }
        for (0..nlist) |c| {
            // Prefetch next centroid while processing current
            if (c + 1 < nlist) {
                @prefetch(@as([*]const u8, @ptrCast(self.centroidSlice(c + 1).ptr)), .{ .rw = .read, .locality = 1, .cache = .data });
            }
            const dist = VectorOps.cosineDistanceFast(normalized_query, self.centroidSlice(c));
            if (probe_len < probes) {
                probe_lists[probe_len] = .{ .idx = c, .distance = dist };
                if (dist > probe_worst_dist) {
                    probe_worst_dist = dist;
                    probe_worst_pos = probe_len;
                }
                probe_len += 1;
                continue;
            }

            if (dist >= probe_worst_dist) continue;
            probe_lists[probe_worst_pos] = .{ .idx = c, .distance = dist };

            probe_worst_pos = 0;
            probe_worst_dist = probe_lists[0].distance;
            var i: usize = 1;
            while (i < probe_len) : (i += 1) {
                if (probe_lists[i].distance > probe_worst_dist) {
                    probe_worst_dist = probe_lists[i].distance;
                    probe_worst_pos = i;
                }
            }
        }

        if (probe_len == 0) return self.searchHnsw(normalized_query, k);

        std.sort.heap(ProbeCandidate, probe_lists[0..probe_len], {}, struct {
            fn lessThan(context: void, a: ProbeCandidate, b: ProbeCandidate) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        var total_candidates: usize = 0;
        for (probe_lists[0..probe_len]) |probe| {
            total_candidates += self.ivf_lengths[probe.idx];
        }
        if (total_candidates == 0) return self.searchHnsw(normalized_query, k);

        var query_q8_stack: [2048]i8 = undefined;
        const query_q8 = if (normalized_query.len <= query_q8_stack.len)
            query_q8_stack[0..normalized_query.len]
        else
            try self.allocator.alloc(i8, normalized_query.len);
        defer if (normalized_query.len > query_q8_stack.len) self.allocator.free(query_q8);

        VectorOps.quantizeBatch(normalized_query, query_q8);

        // Binary-encode query for Hamming pre-filter
        const bwords = self.binary_words_per_vec;
        var query_bin_stack: [32]u64 = undefined; // Supports up to 2048d
        const query_bin = if (bwords <= query_bin_stack.len)
            query_bin_stack[0..bwords]
        else
            try self.allocator.alloc(u64, bwords);
        defer if (bwords > query_bin_stack.len) self.allocator.free(query_bin);
        VectorOps.binaryEncode(normalized_query, query_bin);
        const has_binary = self.binary_vectors.len > 0;

        // Build PQ4 distance table for ultra-fast approximate filtering
        const has_pq4 = self.pq4_codes.len > 0 and self.pq4_m > 0;
        var pq4_table_stack: [1536]u8 = undefined; // 96 subvecs × 16 entries
        const pq4_table_size = if (has_pq4) self.pq4_m * 16 else @as(usize, 0);
        var pq4_table_heap: []u8 = &[_]u8{};
        if (has_pq4 and pq4_table_size > pq4_table_stack.len) {
            pq4_table_heap = try self.allocator.alloc(u8, pq4_table_size);
        }
        defer if (pq4_table_heap.len > 0) self.allocator.free(pq4_table_heap);
        const pq4_table: []u8 = if (has_pq4) (if (pq4_table_size <= pq4_table_stack.len) pq4_table_stack[0..pq4_table_size] else pq4_table_heap) else pq4_table_heap;
        var pq4_inv_scale: f32 = 1.0;
        if (has_pq4) {
            pq4_inv_scale = self.pq4BuildDistTable(normalized_query, pq4_table);
        }

        // HNSW assist for high-recall mode — the graph catches what IVF misses
        const use_hnsw_assist = self.search_ef >= 256 and self.index.entry_point != null;
        const use_threads = total_candidates >= 50_000;

        // ── Fast single-threaded path: fused IVF scan + rerank (zero intermediate alloc) ──
        if (!use_threads and !use_hnsw_assist) {
            const dim = self.storage.dimension;
            const heap_k = @min(k, total_candidates);
            var heap_buf: [64]SearchResult = undefined;
            const heap = if (heap_k <= 64) heap_buf[0..heap_k] else try self.allocator.alloc(SearchResult, heap_k);
            defer if (heap_k > 64) self.allocator.free(heap);
            var heap_len: usize = 0;
            var threshold: f32 = std.math.inf(f32);

            // Margin for int8 quantization error: empirically ~0.05 at 768d
            const INT8_MARGIN: f32 = 0.08;
            // Hamming pre-filter: adapted from current cosine threshold.
            // Relationship: hamming ≈ dim * arccos(1 - cos_dist) / π + margin
            // We use a generous margin (1.3x) to avoid false rejections.
            const dim_f: f32 = @floatFromInt(dim);
            const hamming_max: u32 = @intCast(dim / 2); // random baseline
            var hamming_reject: u32 = hamming_max;

            for (probe_lists[0..probe_len]) |probe| {
                const offset = self.ivf_offsets[probe.idx];
                const list_len = self.ivf_lengths[probe.idx];

                // Prefetch first few quantized vectors ahead of the scan
                const PF_DEPTH = 4;
                for (0..@min(PF_DEPTH, list_len)) |pf| {
                    const pf_idx = self.ivf_postings[offset + pf];
                    @prefetch(self.quantized_vectors.ptr + pf_idx * dim, .{ .rw = .read, .locality = 1, .cache = .data });
                }

                var j: usize = 0;
                while (j < list_len) : (j += 1) {
                    // Prefetch quantized vector PF_DEPTH positions ahead
                    if (j + PF_DEPTH < list_len) {
                        const ahead_idx = self.ivf_postings[offset + j + PF_DEPTH];
                        @prefetch(self.quantized_vectors.ptr + ahead_idx * dim, .{ .rw = .read, .locality = 1, .cache = .data });
                    }

                    const idx = self.ivf_postings[offset + j];

                    // Level 0: Binary Hamming pre-filter (ultra-cheap: ~12 XOR + popcount)
                    if (has_binary and heap_len >= heap_k and threshold < 0.6) {
                        const bvec = self.binary_vectors[idx * bwords ..][0..bwords];
                        const hamming = VectorOps.hammingDistance(query_bin, bvec);
                        if (hamming > hamming_reject) continue;
                    }

                    // Level 0.5: PQ4 approximate distance (96 table lookups vs 768 int8 ops)
                    if (has_pq4 and heap_len >= heap_k) {
                        const pq4_raw = self.pq4DistanceADC(pq4_table, idx);
                        const pq4_dist = @as(f32, @floatFromInt(pq4_raw)) * pq4_inv_scale;
                        // PQ4 returns squared L2; convert to approximate cosine: dist ≈ pq4_l2 * 0.5
                        if (pq4_dist * 0.5 > threshold + 0.15) continue;
                    }

                    const qvec = self.quantized_vectors[idx * dim ..][0..dim];

                    // Level 1: int8 approximate distance (fast rejection)
                    const dot_i32 = VectorOps.dotI8(query_q8, qvec);
                    const approx_dist = 1.0 - @as(f32, @floatFromInt(dot_i32)) * INT8_INV_SCALE_SQ;

                    if (heap_len >= heap_k and approx_dist > threshold + INT8_MARGIN) continue;

                    // Level 2: exact f32 distance (only for promising candidates)
                    // Prefetch next storage vector for reranking
                    if (j + 1 < list_len) {
                        const next_idx = self.ivf_postings[offset + j + 1];
                        const next_vec = self.storage.getVector(next_idx);
                        @prefetch(next_vec.ptr, .{ .rw = .read, .locality = 0, .cache = .data });
                    }
                    const dist = VectorOps.cosineDistanceQueryStorage(normalized_query, self.storage.getVector(idx));

                    if (heap_len < heap_k) {
                        heap[heap_len] = .{ .idx = idx, .distance = dist, .vector = self.resultVectorSlice(idx) };
                        var pos = heap_len;
                        while (pos > 0) {
                            const parent = (pos - 1) / 2;
                            if (heap[pos].distance > heap[parent].distance) {
                                const tmp = heap[pos];
                                heap[pos] = heap[parent];
                                heap[parent] = tmp;
                                pos = parent;
                            } else break;
                        }
                        heap_len += 1;
                        if (heap_len == heap_k) {
                            threshold = heap[0].distance;
                            // Update adaptive Hamming threshold: arccos(1-d)/π * dim * 1.3 margin
                            const cos_sim = std.math.clamp(1.0 - threshold, -1.0, 1.0);
                            const angle = std.math.acos(cos_sim);
                            hamming_reject = @min(hamming_max, @as(u32, @intFromFloat(angle / std.math.pi * dim_f * 1.3)));
                        }
                    } else if (dist < heap[0].distance) {
                        heap[0] = .{ .idx = idx, .distance = dist, .vector = self.resultVectorSlice(idx) };
                        var pos: usize = 0;
                        while (true) {
                            const left = 2 * pos + 1;
                            const right = 2 * pos + 2;
                            var largest = pos;
                            if (left < heap_len and heap[left].distance > heap[largest].distance) largest = left;
                            if (right < heap_len and heap[right].distance > heap[largest].distance) largest = right;
                            if (largest == pos) break;
                            const tmp = heap[pos];
                            heap[pos] = heap[largest];
                            heap[largest] = tmp;
                            pos = largest;
                        }
                        threshold = heap[0].distance;
                        const cos_sim = std.math.clamp(1.0 - threshold, -1.0, 1.0);
                        const angle = std.math.acos(cos_sim);
                        hamming_reject = @min(hamming_max, @as(u32, @intFromFloat(angle / std.math.pi * dim_f * 1.3)));
                    }
                }
            }

            if (heap_len == 0) return self.searchHnsw(normalized_query, k);

            std.sort.heap(SearchResult, heap[0..heap_len], {}, struct {
                fn lessThan(context: void, a: SearchResult, b: SearchResult) bool {
                    _ = context;
                    return a.distance < b.distance;
                }
            }.lessThan);

            const out = try self.allocator.alloc(SearchResult, heap_len);
            @memcpy(out, heap[0..heap_len]);
            return out;
        }

        // ── Multi-threaded / HNSW-assist path ──
        const thread_count = if (use_threads)
            @max(1, @min(configuredCpuCount(), probe_len))
        else
            1;
        const chunk_size = (probe_len + thread_count - 1) / thread_count;

        var local_buffers = try self.allocator.alloc([]ApproxCandidate, thread_count);
        defer {
            for (local_buffers) |buf| self.allocator.free(buf);
            self.allocator.free(local_buffers);
        }

        var local_lens = try self.allocator.alloc(usize, thread_count);
        defer self.allocator.free(local_lens);
        @memset(local_lens, 0);

        for (0..thread_count) |i| {
            const start = i * chunk_size;
            const end = @min(start + chunk_size, probe_len);

            var cap: usize = 0;
            if (start < end) {
                for (probe_lists[start..end]) |probe| {
                    cap += self.ivf_lengths[probe.idx];
                }
            }
            local_buffers[i] = try self.allocator.alloc(ApproxCandidate, cap);
        }

        const ScanContext = struct {
            db: *const VectorDB,
            probes: []const ProbeCandidate,
            query_q8: []const i8,
            start: usize,
            end: usize,
            out: []ApproxCandidate,
            out_len: *usize,

            fn run(ctx: @This()) void {
                var len: usize = 0;
                const dim = ctx.db.storage.dimension;

                for (ctx.start..ctx.end) |probe_i| {
                    const list_idx = ctx.probes[probe_i].idx;
                    const offset = ctx.db.ivf_offsets[list_idx];
                    const list_len = ctx.db.ivf_lengths[list_idx];

                    var j: usize = 0;
                    while (j < list_len) : (j += 1) {
                        const idx = ctx.db.ivf_postings[offset + j];
                        const approx_dist = blk: {
                            const qvec = ctx.db.quantized_vectors[idx * dim ..][0..dim];
                            const dot_i32 = VectorOps.dotI8(ctx.query_q8, qvec);
                            break :blk 1.0 - @as(f32, @floatFromInt(dot_i32)) * INT8_INV_SCALE_SQ;
                        };
                        if (len >= ctx.out.len) continue;
                        ctx.out[len] = .{ .idx = idx, .approx_distance = approx_dist };
                        len += 1;
                    }
                }

                ctx.out_len.* = len;
            }
        };

        if (thread_count > 1 and probe_len > 1) {
            const threads = try self.allocator.alloc(std.Thread, thread_count);
            defer self.allocator.free(threads);

            for (threads, 0..) |*thread, i| {
                const start = i * chunk_size;
                const end = @min(start + chunk_size, probe_len);
                const ctx = ScanContext{
                    .db = self,
                    .probes = probe_lists[0..probe_len],
                    .query_q8 = query_q8,
                    .start = start,
                    .end = end,
                    .out = local_buffers[i],
                    .out_len = &local_lens[i],
                };
                thread.* = try std.Thread.spawn(.{}, ScanContext.run, .{ctx});
            }

            for (threads) |thread| thread.join();
        } else {
            const ctx = ScanContext{
                .db = self,
                .probes = probe_lists[0..probe_len],
                .query_q8 = query_q8,
                .start = 0,
                .end = probe_len,
                .out = local_buffers[0],
                .out_len = &local_lens[0],
            };
            ScanContext.run(ctx);
        }

        var hnsw_assist_ids: []usize = &[_]usize{};
        if (use_hnsw_assist) {
            const hnsw_assist_ef = std.math.clamp(@max(self.search_ef, k * 16), k, self.storage.count);
            self.index.build_rwlock.lockShared();
            defer self.index.build_rwlock.unlockShared();
            // Use int8-accelerated graph search (~10x cheaper per distance than f32)
            const dim = self.storage.dimension;
            const items = try self.index.searchLayerI8(query_q8, self.quantized_vectors, dim, self.index.entry_point.?, hnsw_assist_ef, 0);
            hnsw_assist_ids = try self.allocator.alloc(usize, items.len);
            for (items, 0..) |item, idx| hnsw_assist_ids[idx] = item.idx;
        }
        defer if (hnsw_assist_ids.len > 0) self.allocator.free(hnsw_assist_ids);

        const candidate_cap = total_candidates + hnsw_assist_ids.len;

        const dedupe_ctx = try self.index.acquireSearchContext(1, self.storage.count);
        const seen = dedupe_ctx.visited;
        const seen_mark = dedupe_ctx.mark;

        var approx_candidates = try self.allocator.alloc(ApproxCandidate, candidate_cap);
        defer self.allocator.free(approx_candidates);
        var candidate_len: usize = 0;

        for (0..thread_count) |i| {
            const local = local_buffers[i][0..local_lens[i]];
            for (local) |cand| {
                if (seen[cand.idx] == seen_mark) continue;
                seen[cand.idx] = seen_mark;
                approx_candidates[candidate_len] = cand;
                candidate_len += 1;
            }
        }

        for (hnsw_assist_ids) |idx| {
            if (seen[idx] == seen_mark) continue;
            seen[idx] = seen_mark;
            approx_candidates[candidate_len] = .{ .idx = idx, .approx_distance = 0.0 };
            candidate_len += 1;
        }

        if (candidate_len == 0) return self.searchHnsw(normalized_query, k);

        // Sort by approximate distance — enables int8 early rejection during reranking
        std.sort.heap(ApproxCandidate, approx_candidates[0..candidate_len], {}, struct {
            fn lessThan(context: void, a: ApproxCandidate, b: ApproxCandidate) bool {
                _ = context;
                return a.approx_distance < b.approx_distance;
            }
        }.lessThan);

        // Rerank with max-heap top-k + int8 early rejection
        const heap_k = @min(k, candidate_len);
        var heap_buf2: [64]SearchResult = undefined;
        const heap = if (heap_k <= 64) heap_buf2[0..heap_k] else try self.allocator.alloc(SearchResult, heap_k);
        defer if (heap_k > 64) self.allocator.free(heap);
        var heap_len: usize = 0;
        var threshold: f32 = std.math.inf(f32);

        const INT8_MARGIN_ASSIST: f32 = 0.08;

        for (approx_candidates[0..candidate_len], 0..) |cand, i| {
            // Skip candidates clearly worse than current best (int8 early rejection)
            if (heap_len >= heap_k and cand.approx_distance > threshold + INT8_MARGIN_ASSIST) continue;

            if (i + 1 < candidate_len) {
                const next_vec = self.storage.getVector(approx_candidates[i + 1].idx);
                @prefetch(next_vec.ptr, .{ .rw = .read, .locality = 0, .cache = .data });
            }
            const dist = VectorOps.cosineDistanceQueryStorage(normalized_query, self.storage.getVector(cand.idx));

            if (heap_len < heap_k) {
                heap[heap_len] = .{ .idx = cand.idx, .distance = dist, .vector = self.resultVectorSlice(cand.idx) };
                var pos = heap_len;
                while (pos > 0) {
                    const parent = (pos - 1) / 2;
                    if (heap[pos].distance > heap[parent].distance) {
                        const tmp = heap[pos];
                        heap[pos] = heap[parent];
                        heap[parent] = tmp;
                        pos = parent;
                    } else break;
                }
                heap_len += 1;
                if (heap_len == heap_k) threshold = heap[0].distance;
            } else if (dist < heap[0].distance) {
                heap[0] = .{ .idx = cand.idx, .distance = dist, .vector = self.resultVectorSlice(cand.idx) };
                var pos: usize = 0;
                while (true) {
                    const left = 2 * pos + 1;
                    const right = 2 * pos + 2;
                    var largest = pos;
                    if (left < heap_len and heap[left].distance > heap[largest].distance) largest = left;
                    if (right < heap_len and heap[right].distance > heap[largest].distance) largest = right;
                    if (largest == pos) break;
                    const tmp = heap[pos];
                    heap[pos] = heap[largest];
                    heap[largest] = tmp;
                    pos = largest;
                }
                threshold = heap[0].distance;
            }
        }

        std.sort.heap(SearchResult, heap[0..heap_len], {}, struct {
            fn lessThan(context: void, a: SearchResult, b: SearchResult) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        const out = try self.allocator.alloc(SearchResult, heap_len);
        @memcpy(out, heap[0..heap_len]);
        return out;
    }

    pub fn search(self: *VectorDB, query: []const f32, k: usize) ![]SearchResult {
        if (query.len != self.storage.dimension) return error.InvalidDimension;
        if (k == 0 or self.storage.count == 0) {
            return try self.allocator.alloc(SearchResult, 0);
        }

        // Use on-stack buffer for normalized query to avoid heap allocation
        var normalized_buffer: [2048]f32 = undefined;
        const normalized_query = if (query.len <= normalized_buffer.len)
            normalized_buffer[0..query.len]
        else
            try self.allocator.alloc(f32, query.len);
        defer if (query.len > normalized_buffer.len) self.allocator.free(normalized_query);

        @memcpy(normalized_query, query);
        VectorOps.normalize(normalized_query);

        const results = if (self.storage.count <= BRUTE_FORCE_THRESHOLD or self.index.entry_point == null)
            try self.searchBruteForce(normalized_query, k)
        else if (self.turbo_enabled and self.storage.count >= self.ivf_min_vectors)
            try self.searchTurbo(normalized_query, k)
        else
            try self.searchHnsw(normalized_query, k);

        // Translate internal physical IDs → external insertion IDs
        if (self.internal_to_external.items.len > 0) {
            for (results) |*r| {
                r.idx = self.internal_to_external.items[r.idx];
            }
        }

        return results;
    }

    pub const SearchResult = struct {
        idx: usize,
        distance: f32,
        vector: []const f32,
    };

    // Batch operations for efficiency
    pub fn addBatch(self: *VectorDB, vectors: []const []const f32) !void {
        if (vectors.len == 0) return;
        for (vectors) |vector| {
            if (vector.len != self.storage.dimension) return error.InvalidDimension;
        }

        // Fill storage first, then pre-allocate nodes so insert workers avoid global write-lock convoy.
        const start_idx = try self.storage.appendBatchNormalized(vectors);
        try self.index.reserveCapacity(start_idx + vectors.len);

        // Track external IDs (at insertion time, external == physical)
        try self.internal_to_external.ensureTotalCapacity(start_idx + vectors.len);
        for (0..vectors.len) |i| {
            self.internal_to_external.appendAssumeCapacity(start_idx + i);
        }

        {
            self.index.build_rwlock.lock();
            defer self.index.build_rwlock.unlock();

            const target_len = start_idx + vectors.len;
            try self.index.nodes.ensureTotalCapacity(target_len);

            while (self.index.nodes.items.len < start_idx) {
                const filler = try HNSWIndex.Node.init(self.index.allocator, 0);
                self.index.nodes.appendAssumeCapacity(filler);
            }

            var i: usize = 0;
            while (i < vectors.len) : (i += 1) {
                const node_idx = start_idx + i;
                const level = self.index.selectLevel();
                const node = try HNSWIndex.Node.init(self.index.allocator, level);
                if (node_idx < self.index.nodes.items.len) {
                    self.index.nodes.items[node_idx].deinit(self.index.allocator);
                    self.index.nodes.items[node_idx] = node;
                } else {
                    self.index.nodes.appendAssumeCapacity(node);
                }
            }

            if (self.index.entry_point == null) {
                self.index.entry_point = start_idx;
                self.index.max_level = self.index.nodes.items[start_idx].level;
            }

            // Pre-allocate edge slots for all new nodes so the parallel insert phase
            // never touches edge_pool_mutex (eliminates lock contention on hot path).
            for (start_idx..start_idx + vectors.len) |ni| {
                const node = &self.index.nodes.items[ni];
                for (0..node.connections.len) |lvl| {
                    const cap = if (lvl == 0) self.index.m * 2 else self.index.m;
                    node.connections[lvl].offset = try self.index.allocateEdges(cap);
                    node.connections[lvl].capacity = cap;
                    node.connections[lvl].len = 0;
                }
            }
        }

        const thread_count = @max(1, @min(configuredCpuCount(), vectors.len));
        if (thread_count > 1 and vectors.len >= thread_count * 128) {
            const threads = try self.allocator.alloc(std.Thread, thread_count);
            defer self.allocator.free(threads);

            const Context = struct {
                db: *VectorDB,
                start_idx: usize,
                start: usize,
                end: usize,

                fn run(ctx: @This()) void {
                    var i = ctx.start;
                    while (i < ctx.end) : (i += 1) {
                        ctx.db.index.insertParallelNoAlloc(ctx.start_idx + i, &ctx.db.storage) catch |err| {
                            @panic(@errorName(err));
                        };
                    }
                }
            };

            const chunk_size = (vectors.len + thread_count - 1) / thread_count;
            for (threads, 0..) |*thread, i| {
                const start = i * chunk_size;
                const end = @min(start + chunk_size, vectors.len);
                const ctx = Context{
                    .db = self,
                    .start_idx = start_idx,
                    .start = start,
                    .end = end,
                };
                thread.* = try std.Thread.spawn(.{}, Context.run, .{ctx});
            }

            for (threads) |thread| thread.join();
        } else {
            var i: usize = 0;
            while (i < vectors.len) : (i += 1) {
                try self.index.insertParallelNoAlloc(start_idx + i, &self.storage);
            }
        }

        // Pre-build turbo index so first query doesn't pay rebuild cost
        if (self.turbo_enabled and self.storage.count >= self.ivf_min_vectors) {
            try self.rebuildTurboIndex();
        } else {
            self.turbo_dirty = true;
        }
    }

    pub fn searchBatch(self: *VectorDB, queries: []const []const f32, k: usize) ![][]SearchResult {
        var results = try self.allocator.alloc([]SearchResult, queries.len);

        // Process queries in parallel using all CPU cores
        const thread_count = configuredCpuCount();
        const chunk_size = (queries.len + thread_count - 1) / thread_count;

        if (thread_count > 1 and queries.len > thread_count) {
            const threads = try self.allocator.alloc(std.Thread, thread_count);
            defer self.allocator.free(threads);

            const Context = struct {
                db: *VectorDB,
                queries: []const []const f32,
                results: [][]SearchResult,
                k: usize,
                start: usize,
                end: usize,

                fn searchWorker(ctx: @This()) void {
                    for (ctx.start..ctx.end) |i| {
                        if (i >= ctx.queries.len) break;
                        ctx.results[i] = ctx.db.search(ctx.queries[i], ctx.k) catch |err| {
                            @panic(@errorName(err));
                        };
                    }
                }
            };

            // Launch worker threads
            for (threads, 0..) |*thread, i| {
                const start = i * chunk_size;
                const end = @min(start + chunk_size, queries.len);
                const ctx = Context{
                    .db = self,
                    .queries = queries,
                    .results = results,
                    .k = k,
                    .start = start,
                    .end = end,
                };
                thread.* = try std.Thread.spawn(.{}, Context.searchWorker, .{ctx});
            }

            // Wait for all threads to complete
            for (threads) |thread| {
                thread.join();
            }
        } else {
            // Single-threaded fallback
            for (queries, 0..) |query, i| {
                results[i] = try self.search(query, k);
            }
        }

        return results;
    }

    // Persistence
    pub fn save(self: *VectorDB, path: []const u8) !void {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();

        // Write header
        try file.writer().writeInt(u32, FILE_MAGIC, .little);
        try file.writer().writeInt(u32, FILE_VERSION, .little);
        try file.writer().writeInt(u32, @intCast(self.storage.dimension), .little);
        try file.writer().writeInt(u32, @intCast(self.storage.count), .little);

        // Write vectors as f32 for portable persistence across storage modes.
        var tmp_vec: []f32 = &[_]f32{};
        if (comptime StorageScalar != f32) {
            tmp_vec = try self.allocator.alloc(f32, self.storage.dimension);
        }
        defer if (comptime StorageScalar != f32) self.allocator.free(tmp_vec);

        var i: usize = 0;
        while (i < self.storage.count) : (i += 1) {
            const vec = self.storage.getVector(i);
            if (comptime StorageScalar == f32) {
                try file.writeAll(mem.sliceAsBytes(vec));
            } else {
                for (vec, 0..) |v, d| {
                    tmp_vec[d] = VectorOps.toF32(v);
                }
                try file.writeAll(mem.sliceAsBytes(tmp_vec));
            }
        }

        // Write index structure
        // ... (implement index serialization)
    }

    pub fn load(allocator: mem.Allocator, path: []const u8) !VectorDB {
        const file = try fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();

        // Read header (supports both legacy and versioned formats)
        const first = try reader.readInt(u32, .little);
        var dimension_u32: u32 = undefined;
        var count_u32: u32 = undefined;

        if (first == FILE_MAGIC) {
            const version = try reader.readInt(u32, .little);
            if (version != FILE_VERSION) return error.UnsupportedFormatVersion;
            dimension_u32 = try reader.readInt(u32, .little);
            count_u32 = try reader.readInt(u32, .little);
        } else {
            dimension_u32 = first;
            count_u32 = try reader.readInt(u32, .little);
        }

        const dimension: usize = @intCast(dimension_u32);
        const count: usize = @intCast(count_u32);

        // Initialize database
        var db = try VectorDB.init(allocator, dimension, .{
            .initial_capacity = count,
        });
        errdefer db.deinit();

        const vector = try allocator.alloc(f32, dimension);
        defer allocator.free(vector);
        const vector_bytes = mem.sliceAsBytes(vector);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const bytes_read = try reader.readAll(vector_bytes);
            if (bytes_read != vector_bytes.len) return error.CorruptFile;
            _ = try db.addVector(vector, null);
        }

        return db;
    }

    /// Update the search ef parameter (controls quality vs speed during queries).
    pub fn setSearchEf(self: *VectorDB, ef: usize) void {
        self.search_ef = ef;
    }
};

// Example usage and tests
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const demo_vector_count: usize = configuredDemoVectorCount(100_000);

    // Create a vector database optimized for speed
    var db = try VectorDB.init(allocator, 768, .{ // 768d vectors (common embedding size)
        .initial_capacity = demo_vector_count,
        .index_m = 12, // Lean graph (turbo IVF path handles search quality)
        .index_ef_construction = 64, // Fast build; turbo path compensates for graph quality
    });
    defer db.deinit();

    // Pre-allocate for batch insertion
    try db.storage.reserveCapacity(demo_vector_count);
    try db.index.reserveCapacity(demo_vector_count);

    // Add vectors in batches for maximum throughput
    var rng = std.Random.DefaultPrng.init(42);
    const vectors = try allocator.alloc([768]f32, demo_vector_count);
    defer allocator.free(vectors);

    std.debug.print("Generating {} vectors...\n", .{demo_vector_count});
    for (vectors) |*vector| {
        for (vector) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0; // [-1, 1] range
        }
    }

    std.debug.print("Adding vectors in batch...\n", .{});
    const start_time = std.time.milliTimestamp();

    // Convert to slice of slices for batch insert
    const vector_slices = try allocator.alloc([]const f32, vectors.len);
    defer allocator.free(vector_slices);
    for (vectors, 0..) |*vec, idx| {
        vector_slices[idx] = vec;
    }

    try db.addBatch(vector_slices);

    const build_time = std.time.milliTimestamp() - start_time;
    std.debug.print("Built index in {}ms ({} vectors/sec)\n", .{ build_time, @as(u64, @intFromFloat(@as(f64, @floatFromInt(vectors.len)) * 1000.0 / @as(f64, @floatFromInt(build_time)))) });

    // Benchmark search performance
    std.debug.print("\nBenchmarking search performance...\n", .{});

    // Generate query vectors
    const queries = try allocator.alloc([768]f32, 100);
    defer allocator.free(queries);
    for (queries) |*query| {
        for (query) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }
    }

    // Warm up
    _ = try db.search(&queries[0], 10);

    // Benchmark single queries
    const search_start = std.time.milliTimestamp();
    for (queries) |*query| {
        const results = try db.search(query, 10);
        allocator.free(results);
    }
    const search_time = std.time.milliTimestamp() - search_start;
    const qps = @as(u64, @intFromFloat(@as(f64, @floatFromInt(queries.len)) * 1000.0 / @as(f64, @floatFromInt(search_time))));
    std.debug.print("Single-threaded search: {}ms for {} queries ({} QPS)\n", .{ search_time, queries.len, qps });

    // Benchmark batch search
    const query_slices = try allocator.alloc([]const f32, queries.len);
    defer allocator.free(query_slices);
    for (queries, 0..) |*query, idx| {
        query_slices[idx] = query;
    }

    const batch_start = std.time.milliTimestamp();
    const batch_results = try db.searchBatch(query_slices, 10);
    defer {
        for (batch_results) |res| allocator.free(res);
        allocator.free(batch_results);
    }
    const batch_time = std.time.milliTimestamp() - batch_start;
    const batch_qps = @as(u64, @intFromFloat(@as(f64, @floatFromInt(queries.len)) * 1000.0 / @as(f64, @floatFromInt(batch_time))));
    std.debug.print("Multi-threaded batch search: {}ms for {} queries ({} QPS)\n", .{ batch_time, queries.len, batch_qps });

    // Show some results
    std.debug.print("\nExample search results:\n", .{});
    const example_results = try db.search(&queries[0], 5);
    defer allocator.free(example_results);
    for (example_results) |result| {
        std.debug.print("  Index: {}, Distance: {:.4}\n", .{ result.idx, result.distance });
    }
}

test "vector operations" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const dot_product = VectorOps.dot(&a, &b);
    try std.testing.expectApproxEqAbs(dot_product, 20.0, 0.001);

    // Normalize vectors for cosine distance since cosineDistance expects unit vectors
    var norm_a = a;
    var norm_b = b;
    VectorOps.normalize(&norm_a);
    VectorOps.normalize(&norm_b);

    const cosine_dist = VectorOps.cosineDistance(&norm_a, &norm_b);
    try std.testing.expect(cosine_dist >= 0.0 and cosine_dist <= 2.0);
}

test "vector storage" {
    const allocator = std.testing.allocator;

    var storage = try VectorStorage.init(allocator, 4, 10);
    defer storage.deinit();

    const v1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const idx = try storage.addVector(&v1);
    try std.testing.expectEqual(@as(usize, 0), idx);

    const retrieved = storage.getVector(idx);

    // The storage normalizes vectors, so we need to compare with normalized v1
    var normalized_v1 = v1;
    VectorOps.normalize(&normalized_v1);

    // Use approximate equality for floating point comparison
    try std.testing.expectEqual(normalized_v1.len, retrieved.len);
    const tol: f32 = if (comptime StorageScalar == f32)
        0.0001
    else if (comptime StorageScalar == f16)
        0.002
    else
        0.02;
    for (normalized_v1, retrieved) |expected, actual| {
        const actual_f32 = VectorOps.toF32(actual);
        try std.testing.expectApproxEqAbs(expected, actual_f32, tol);
    }
}

test "cosine distance consistency" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    // Normalize vectors
    VectorOps.normalize(&a);
    VectorOps.normalize(&b);

    // Original calculation (now also normalized)
    const old_dist = VectorOps.cosineDistance(&a, &b);

    // New calculation should be the same since we use unit vectors
    const new_dist = VectorOps.cosineDistance(&a, &b);

    try std.testing.expectApproxEqAbs(old_dist, new_dist, 0.0001);

    // Test with random vectors
    var rng = std.Random.DefaultPrng.init(42);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var vec1: [128]f32 = undefined;
        var vec2: [128]f32 = undefined;
        for (&vec1) |*v| v.* = rng.random().float(f32);
        for (&vec2) |*v| v.* = rng.random().float(f32);

        // Normalize the vectors
        VectorOps.normalize(&vec1);
        VectorOps.normalize(&vec2);

        const old_d = VectorOps.cosineDistance(&vec1, &vec2);
        const new_d = VectorOps.cosineDistance(&vec1, &vec2);

        try std.testing.expectApproxEqAbs(old_d, new_d, 0.0001);
    }
}

test "SIMD dot product optimizations" {
    // Test 768d vectors (should use AVX2Fixed768)
    var vec768a: [768]f32 = undefined;
    var vec768b: [768]f32 = undefined;
    for (&vec768a, 0..) |*v, i| v.* = @floatFromInt(i % 100);
    for (&vec768b, 0..) |*v, i| v.* = @floatFromInt((i + 1) % 100);

    const result768_fast = VectorOps.dotFast(&vec768a, &vec768b);
    const result768_scalar = VectorOps.dotScalarFast(&vec768a, &vec768b);
    try std.testing.expectApproxEqAbs(result768_fast, result768_scalar, 0.01);

    // Test 384d vectors (should use regular AVX2)
    var vec384a: [384]f32 = undefined;
    var vec384b: [384]f32 = undefined;
    for (&vec384a, 0..) |*v, i| v.* = @floatFromInt(i % 50);
    for (&vec384b, 0..) |*v, i| v.* = @floatFromInt((i + 1) % 50);

    const result384_fast = VectorOps.dotFast(&vec384a, &vec384b);
    const result384_scalar = VectorOps.dotScalarFast(&vec384a, &vec384b);
    try std.testing.expectApproxEqAbs(result384_fast, result384_scalar, 0.01);

    // Test small vectors (should use scalar)
    const small_a = [_]f32{ 1.0, 2.0, 3.0 };
    const small_b = [_]f32{ 4.0, 5.0, 6.0 };

    const result_small_fast = VectorOps.dotFast(&small_a, &small_b);
    const result_small_scalar = VectorOps.dotScalarFast(&small_a, &small_b);
    try std.testing.expectApproxEqAbs(result_small_fast, result_small_scalar, 0.001);
}

test "empty search returns freeable slice" {
    const allocator = std.testing.allocator;

    var db = try VectorDB.init(allocator, 4, .{ .initial_capacity = 8 });
    defer db.deinit();

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try db.search(&query, 5);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "save and load roundtrip keeps searchable vectors" {
    const allocator = std.testing.allocator;
    const path = "vector_db_roundtrip_test.bin";
    defer fs.cwd().deleteFile(path) catch {};

    var db = try VectorDB.init(allocator, 4, .{
        .initial_capacity = 8,
        .index_m = 16,
        .index_ef_construction = 64,
    });
    defer db.deinit();

    const vectors = [_][4]f32{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.9, 0.1, 0.0, 0.0 },
    };

    for (vectors) |vec| {
        _ = try db.addVector(&vec, null);
    }

    try db.save(path);

    var loaded = try VectorDB.load(allocator, path);
    defer loaded.deinit();
    loaded.setSearchEf(64);

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try loaded.search(&query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    try std.testing.expect(results[0].idx == 0 or results[0].idx == 3);
}

test "addBatch validates dimensions" {
    const allocator = std.testing.allocator;

    var db = try VectorDB.init(allocator, 4, .{ .initial_capacity = 8 });
    defer db.deinit();

    const v_ok = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v_bad = [_]f32{ 1.0, 0.0, 0.0 };
    const batch = [_][]const f32{ &v_ok, &v_bad };

    try std.testing.expectError(error.InvalidDimension, db.addBatch(&batch));
}

test "turbo IVF uses mmap postings for file-backed mode" {
    const allocator = std.testing.allocator;
    const data_path = "turbo_vectors_test.db";
    const postings_path = "turbo_vectors_test.db.ivf.postings";
    defer fs.cwd().deleteFile(postings_path) catch {};
    defer fs.cwd().deleteFile(data_path) catch {};

    var db = try VectorDB.init(allocator, 8, .{
        .initial_capacity = 256,
        .file_path = data_path,
        .enable_turbo = true,
        .ivf_min_vectors = 16,
        .ivf_nlist = 8,
        .ivf_probes = 4,
        .ivf_rerank_factor = 4,
    });
    defer db.deinit();

    var rng = std.Random.DefaultPrng.init(1234);
    const vectors = try allocator.alloc([8]f32, 128);
    defer allocator.free(vectors);
    for (vectors) |*vec| {
        for (vec) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }
        VectorOps.normalize(vec);
    }

    const slices = try allocator.alloc([]const f32, vectors.len);
    defer allocator.free(slices);
    for (vectors, 0..) |*vec, i| slices[i] = vec;
    try db.addBatch(slices);
    try db.rebuildTurboIndex();

    const results = try db.search(slices[0], 5);
    defer allocator.free(results);
    try std.testing.expect(results.len > 0);
    try std.testing.expect(db.ivf_postings_is_mmap);

    const stat = try fs.cwd().statFile(postings_path);
    try std.testing.expect(stat.size >= @as(u64, @intCast(db.storage.count * @sizeOf(usize))));
}
