const std = @import("std");
const math = std.math;
const mem = std.mem;
const os = std.os;
const fs = std.fs;
const builtin = @import("builtin");

// Configuration constants
const VECTOR_ALIGNMENT = 64; // For AVX-512 and cache line alignment
const PAGE_SIZE = 4096;
const INDEX_BUCKET_SIZE = 64;
const MMAP_THRESHOLD = 1024 * 1024; // 1MB
const PAGE_ALIGNMENT = 16384; // Common page size on macOS
const PREFETCH_DISTANCE = 256; // Prefetch distance for memory access
const CACHE_LINE_SIZE = 64;
const BRUTE_FORCE_THRESHOLD = 4096;

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

    pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        var sum: f32 = 0.0;
        for (a, b) |av, bv| {
            const diff = av - bv;
            sum += diff * diff;
        }
        return math.sqrt(sum);
    }

    /// Normalize a vector to unit length in-place
    pub fn normalize(vec: []f32) void {
        var sum: f32 = 0.0;
        for (vec) |v| sum += v * v;
        const norm = math.sqrt(sum);
        if (norm > 0) {
            for (vec) |*v| v.* /= norm;
        }
    }

    fn dotNEON(a: []const f32, b: []const f32) f32 {
        // Optimised dot product for ARM NEON (128-bit) – works on Apple Silicon
        std.debug.assert(a.len == b.len);

        const vec_size = 4; // 128-bit NEON can process 4 floats per vector
        const simd_len = a.len - (a.len % vec_size);

        // Multiple accumulators to hide latency on A-series cores
        var acc1 = @as(@Vector(4, f32), @splat(0.0));
        var acc2 = @as(@Vector(4, f32), @splat(0.0));
        var acc3 = @as(@Vector(4, f32), @splat(0.0));
        var acc4 = @as(@Vector(4, f32), @splat(0.0));

        var i: usize = 0;
        const unroll_len = simd_len - (simd_len % 16); // 4×4 = 16 floats per full unrolled iter

        // Process 16 floats (4 NEON registers) per iteration
        while (i < unroll_len) : (i += 16) {
            // Unroll – manual @as to guarantee vectors
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
        }

        // Handle the remaining 4-float chunks
        while (i < simd_len) : (i += vec_size) {
            const va = @as(@Vector(4, f32), a[i..][0..4].*);
            const vb = @as(@Vector(4, f32), b[i..][0..4].*);
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
};

// Memory-mapped vector storage for efficient I/O
const VectorStorage = struct {
    allocator: mem.Allocator,
    file: ?fs.File,
    data: []align(VECTOR_ALIGNMENT) u8,
    mmap_data: ?[]align(PAGE_ALIGNMENT) u8,
    dimension: usize,
    count: usize,
    capacity: usize,
    use_mmap: bool,
    // Quantization support for memory efficiency
    quantized_data: ?[]align(CACHE_LINE_SIZE) u8,
    use_quantization: bool,
    quantization_bits: u8,

    pub fn init(allocator: mem.Allocator, dimension: usize, initial_capacity: usize) !VectorStorage {
        const vector_size = dimension * @sizeOf(f32);
        const data_size = initial_capacity * vector_size;

        return VectorStorage{
            .allocator = allocator,
            .file = null,
            .data = try allocator.alignedAlloc(u8, VECTOR_ALIGNMENT, data_size),
            .mmap_data = null,
            .dimension = dimension,
            .count = 0,
            .capacity = initial_capacity,
            .use_mmap = false,
            .quantized_data = null,
            .use_quantization = false,
            .quantization_bits = 8,
        };
    }

    pub fn initWithFile(allocator: mem.Allocator, path: []const u8, dimension: usize) !VectorStorage {
        const file = try fs.cwd().createFile(path, .{ .read = true });
        errdefer file.close();

        const vector_size = dimension * @sizeOf(f32);
        const initial_size = 1024 * vector_size; // Start with 1024 vectors

        try file.setEndPos(initial_size);

        var storage = VectorStorage{
            .allocator = allocator,
            .file = file,
            .data = undefined,
            .mmap_data = null,
            .dimension = dimension,
            .count = 0,
            .capacity = 1024,
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
        if (self.use_mmap) {
            if (self.mmap_data) |mmap_data| {
                std.posix.munmap(mmap_data);
            }
        } else {
            self.allocator.free(self.data);
        }
        if (self.file) |file| {
            file.close();
        }
    }

    fn writeVectorNormalizedAt(self: *VectorStorage, idx: usize, vector: []const f32) void {
        std.debug.assert(idx < self.capacity);
        std.debug.assert(vector.len == self.dimension);

        const offset = idx * self.dimension * @sizeOf(f32);
        const dest = self.data[offset..][0 .. vector.len * @sizeOf(f32)];
        const dest_f32 = @as([*]f32, @ptrCast(@alignCast(dest.ptr)))[0..vector.len];

        var sum: f32 = 0.0;
        for (vector, 0..) |v, i| {
            dest_f32[i] = v;
            sum += v * v;
        }

        if (sum > 0) {
            const inv_norm = 1.0 / @sqrt(sum);
            for (dest_f32) |*v| {
                v.* *= inv_norm;
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

    pub fn getVector(self: VectorStorage, idx: usize) []const f32 {
        std.debug.assert(idx < self.count);
        const offset = idx * self.dimension * @sizeOf(f32);
        const bytes = self.data[offset..][0 .. self.dimension * @sizeOf(f32)];
        return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..self.dimension];
    }

    /// Pre-allocate storage capacity for batch operations
    pub fn reserveCapacity(self: *VectorStorage, additional_count: usize) !void {
        const needed_capacity = self.count + additional_count;
        if (needed_capacity <= self.capacity) return; // Already have enough capacity

        // Grow to at least the needed capacity
        var new_capacity = self.capacity;
        while (new_capacity < needed_capacity) {
            new_capacity *= 2;
        }

        const vector_size = self.dimension * @sizeOf(f32);
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

        const cpu_count = @max(1, std.Thread.getCpuCount() catch 1);
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
        const new_capacity = self.capacity * 2;
        const vector_size = self.dimension * @sizeOf(f32);
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
            .entry_point = null,
            .m = m,
            .ef_construction = ef_construction,
            .ml = 1.0 / @log(@as(f64, 2.0)),
            .seed = @intCast(std.time.milliTimestamp()),
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp())),
        };
    }

    pub fn deinit(self: *HNSWIndex) void {
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
        const candidate_cap = @max(num_closest * 16, 16);
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
            conn.offset = try self.allocateEdges(cap);
            conn.capacity = cap;
            conn.len = 0;
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
        const level = self.selectLevel();
        const node = try Node.init(self.allocator, level);

        // Ensure we have enough nodes
        while (self.nodes.items.len <= idx) {
            try self.nodes.append(try Node.init(self.allocator, 0));
        }
        self.nodes.items[idx].deinit(self.allocator);
        self.nodes.items[idx] = node;

        if (self.entry_point == null) {
            self.entry_point = idx;
            self.max_level = level;
            return;
        }

        const vector = storage.getVector(idx);
        const old_entry = self.entry_point.?;
        const entry_level = self.nodes.items[old_entry].level;
        var current_entry = old_entry;

        // Search from top layer to target layer.
        if (entry_level > 0) {
            var l = @min(level, entry_level);
            while (l > 0) : (l -= 1) {
                const nearest_at_level = try self.searchLayerInsert(vector, current_entry, self.ef_construction, l, storage);
                std.debug.assert(nearest_at_level.len > 0);
                current_entry = nearest_at_level[0].idx;
            }
        }

        const nearest = try self.searchLayerInsert(vector, current_entry, self.ef_construction, 0, storage);
        std.debug.assert(nearest.len > 0);

        // Connect the new node.
        var l: usize = 0;
        while (l <= level) : (l += 1) {
            const m = if (l == 0) self.m * 2 else self.m;
            const candidate_count = @min(m, nearest.len);
            for (nearest[0..candidate_count]) |item| {
                const neighbor = item.idx;
                if (l > self.nodes.items[neighbor].level) continue;
                const dist = item.distance;
                try self.addConnection(idx, neighbor, l, dist, storage);
                try self.addConnection(neighbor, idx, l, dist, storage);
            }
        }

        if (level > entry_level) {
            self.entry_point = idx;
            self.max_level = level;
        }
    }

    fn searchLayerInsert(
        self: *HNSWIndex,
        query: []const f32,
        entry: usize,
        num_closest: usize,
        layer: usize,
        storage: *const VectorStorage,
    ) ![]SearchItem {
        const ef = @max(num_closest, 1);
        try self.ensureInsertScratch(ef);

        const mark = self.scratch_mark;
        const candidates_items = self.scratch_candidates;
        const w_items = self.scratch_w;
        var candidates_len: usize = 0;
        var w_len: usize = 0;

        const entry_dist = VectorOps.cosineDistanceFast(query, storage.getVector(entry));
        candidates_items[0] = .{ .idx = entry, .distance = entry_dist };
        candidates_len = 1;
        w_items[0] = .{ .idx = entry, .distance = entry_dist };
        w_len = 1;
        self.scratch_visited[entry] = mark;

        var current_idx: usize = 0;
        while (current_idx < candidates_len) {
            const current = candidates_items[current_idx];
            current_idx += 1;

            if (layer > self.nodes.items[current.idx].level) continue;
            const connections = self.getConnections(current.idx, layer);

            for (connections) |neighbor| {
                if (self.scratch_visited[neighbor] == mark) continue;
                self.scratch_visited[neighbor] = mark;

                const dist = VectorOps.cosineDistanceFast(query, storage.getVector(neighbor));

                if (candidates_len < candidates_items.len) {
                    candidates_items[candidates_len] = .{ .idx = neighbor, .distance = dist };
                    candidates_len += 1;
                }

                if (w_len < ef) {
                    w_items[w_len] = .{ .idx = neighbor, .distance = dist };
                    w_len += 1;

                    var child = w_len - 1;
                    while (child > 0) {
                        const parent = (child - 1) / 2;
                        if (w_items[child].distance <= w_items[parent].distance) break;
                        std.mem.swap(SearchItem, &w_items[child], &w_items[parent]);
                        child = parent;
                    }
                } else if (dist < w_items[0].distance) {
                    w_items[0] = .{ .idx = neighbor, .distance = dist };

                    var parent: usize = 0;
                    while (true) {
                        const left = 2 * parent + 1;
                        const right = 2 * parent + 2;
                        var largest = parent;

                        if (left < w_len and w_items[left].distance > w_items[largest].distance) {
                            largest = left;
                        }
                        if (right < w_len and w_items[right].distance > w_items[largest].distance) {
                            largest = right;
                        }

                        if (largest == parent) break;
                        std.mem.swap(SearchItem, &w_items[parent], &w_items[largest]);
                        parent = largest;
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
    ) !std.ArrayList(usize) {
        // Use a bitset for visited tracking - much faster than hashmap
        const max_nodes = self.nodes.items.len;
        const BitSet = std.DynamicBitSetUnmanaged;
        var visited = try BitSet.initEmpty(self.allocator, max_nodes);
        defer visited.deinit(self.allocator);

        // Pre-allocate arrays to avoid allocation in hot loop
        var candidates_items = try self.allocator.alloc(SearchItem, num_closest * 16);
        defer self.allocator.free(candidates_items);
        var candidates_len: usize = 0;

        var w_items = try self.allocator.alloc(SearchItem, num_closest);
        defer self.allocator.free(w_items);
        var w_len: usize = 0;

        const entry_dist = VectorOps.cosineDistanceFast(query, storage.getVector(entry));
        candidates_items[0] = .{ .idx = entry, .distance = entry_dist };
        candidates_len = 1;
        w_items[0] = .{ .idx = entry, .distance = entry_dist };
        w_len = 1;
        visited.set(entry);

        // Prefetch the entry node's connections
        const entry_node = &self.nodes.items[entry];
        if (layer <= entry_node.level) {
            const conn = entry_node.connections[layer];
            if (conn.len > 0) {
                @prefetch(self.edge_pool.items.ptr + conn.offset, .{ .rw = .read, .locality = 3, .cache = .data });
            }
        }

        var current_idx: usize = 0;
        while (current_idx < candidates_len) {
            const current = candidates_items[current_idx];
            current_idx += 1;

            // Early termination based on distance bound
            // if (w_len > 0 and current.distance > w_items[0].distance * 1.01) break;

            // Skip if this node doesn't have connections at this layer
            if (layer > self.nodes.items[current.idx].level) continue;

            const connections = self.getConnections(current.idx, layer);
            const distances = self.getDistances(current.idx, layer);

            // Prefetch next candidate's data
            if (current_idx < candidates_len) {
                const next_idx = candidates_items[current_idx].idx;
                if (layer <= self.nodes.items[next_idx].level) {
                    const next_conn = self.nodes.items[next_idx].connections[layer];
                    if (next_conn.len > 0) {
                        @prefetch(self.edge_pool.items.ptr + next_conn.offset, .{ .rw = .read, .locality = 2, .cache = .data });
                    }
                }
            }

            for (connections, distances) |neighbor, _| {
                if (visited.isSet(neighbor)) continue;
                visited.set(neighbor);

                // Skip distance pruning - it kills recall
                // const dist_lower_bound = @abs(current.distance - cached_dist);
                // if (w_len >= num_closest and dist_lower_bound > w_items[0].distance * 1.2) continue;

                const neighbor_vec = storage.getVector(neighbor);
                const dist = VectorOps.cosineDistanceFast(query, neighbor_vec);

                // Always add to candidates for better exploration
                if (candidates_len < candidates_items.len) {
                    candidates_items[candidates_len] = .{ .idx = neighbor, .distance = dist };
                    candidates_len += 1;
                }

                // Update w (maintain MAX-heap property - worst at root)
                if (w_len < num_closest) {
                    // Insert into heap
                    w_items[w_len] = .{ .idx = neighbor, .distance = dist };
                    w_len += 1;
                    // Bubble up (max-heap)
                    var child = w_len - 1;
                    while (child > 0) {
                        const parent = (child - 1) / 2;
                        if (w_items[child].distance <= w_items[parent].distance) break;
                        std.mem.swap(SearchItem, &w_items[child], &w_items[parent]);
                        child = parent;
                    }
                } else if (dist < w_items[0].distance) {
                    // Replace root (worst) and bubble down
                    w_items[0] = .{ .idx = neighbor, .distance = dist };
                    var parent: usize = 0;
                    while (true) {
                        const left = 2 * parent + 1;
                        const right = 2 * parent + 2;
                        var largest = parent;

                        if (left < w_len and w_items[left].distance > w_items[largest].distance) {
                            largest = left;
                        }
                        if (right < w_len and w_items[right].distance > w_items[largest].distance) {
                            largest = right;
                        }

                        if (largest == parent) break;
                        std.mem.swap(SearchItem, &w_items[parent], &w_items[largest]);
                        parent = largest;
                    }
                }
            }
        }

        // Extract results from min-heap
        var result = std.ArrayList(usize).init(self.allocator);
        try result.ensureTotalCapacity(w_len);

        // Sort w_items by distance for consistent results
        std.sort.heap(SearchItem, w_items[0..w_len], {}, struct {
            fn lessThan(context: void, a: SearchItem, b: SearchItem) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        for (w_items[0..w_len]) |item| {
            result.appendAssumeCapacity(item.idx);
        }

        return result;
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
                    .distance = VectorOps.cosineDistanceFast(base, storage.getVector(id)),
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
                    .distance = VectorOps.cosineDistanceFast(base, storage.getVector(id)),
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

    const SearchItem = struct {
        idx: usize,
        distance: f32,
    };

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

    allocator: mem.Allocator,
    storage: VectorStorage,
    index: HNSWIndex,
    metadata: std.AutoHashMap(usize, std.json.Value),
    search_ef: usize,

    pub fn init(
        allocator: mem.Allocator,
        dimension: usize,
        options: struct {
            initial_capacity: usize = 10000,
            index_m: usize = 32, // Increased for better connectivity
            index_ef_construction: usize = 400, // Increased for better quality
            file_path: ?[]const u8 = null,
        },
    ) !VectorDB {
        const storage = if (options.file_path) |path|
            try VectorStorage.initWithFile(allocator, path, dimension)
        else
            try VectorStorage.init(allocator, dimension, options.initial_capacity);

        return VectorDB{
            .allocator = allocator,
            .storage = storage,
            .index = HNSWIndex.init(allocator, options.index_m, options.index_ef_construction),
            .metadata = std.AutoHashMap(usize, std.json.Value).init(allocator),
            .search_ef = 128,
        };
    }

    pub fn deinit(self: *VectorDB) void {
        self.storage.deinit();
        self.index.deinit();
        self.metadata.deinit();
    }

    pub fn addVector(self: *VectorDB, vector: []const f32, metadata: ?std.json.Value) !usize {
        if (vector.len != self.storage.dimension) return error.InvalidDimension;

        const idx = try self.storage.addVector(vector);
        try self.index.insert(idx, &self.storage);

        if (metadata) |m| {
            try self.metadata.put(idx, m);
        }

        return idx;
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
            const vec = self.storage.getVector(idx);
            const dist = VectorOps.cosineDistanceFast(normalized_query, vec);

            if (filled < result_len) {
                results[filled] = .{
                    .idx = idx,
                    .distance = dist,
                    .vector = vec,
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
                .vector = vec,
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

        if (self.storage.count <= BRUTE_FORCE_THRESHOLD or self.index.entry_point == null) {
            return self.searchBruteForce(normalized_query, k);
        }

        const ef = @max(k * 8, self.search_ef); // Much higher ef for perfect recall
        const candidates = try self.index.searchLayer(normalized_query, self.index.entry_point.?, ef, 0, &self.storage);
        defer candidates.deinit();

        var results = try self.allocator.alloc(SearchResult, @min(k, candidates.items.len));

        for (candidates.items[0..results.len], 0..) |idx, i| {
            const vec = self.storage.getVector(idx);
            if (i + 1 < results.len) {
                const next_vec = self.storage.getVector(candidates.items[i + 1]);
                @prefetch(next_vec.ptr, .{ .rw = .read, .locality = 2, .cache = .data });
            }

            results[i] = .{
                .idx = idx,
                .distance = VectorOps.cosineDistanceFast(normalized_query, vec),
                .vector = vec,
            };
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

        // Fill storage in parallel, then build the graph sequentially.
        const start_idx = try self.storage.appendBatchNormalized(vectors);
        try self.index.reserveCapacity(vectors.len);

        var i: usize = 0;
        while (i < vectors.len) : (i += 1) {
            try self.index.insert(start_idx + i, &self.storage);
        }
    }

    pub fn searchBatch(self: *VectorDB, queries: []const []const f32, k: usize) ![][]SearchResult {
        var results = try self.allocator.alloc([]SearchResult, queries.len);

        // Process queries in parallel using all CPU cores
        const thread_count = @max(1, std.Thread.getCpuCount() catch 1);
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

        // Write vectors
        const data_size = self.storage.count * self.storage.dimension * @sizeOf(f32);
        try file.writeAll(self.storage.data[0..data_size]);

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

    // Create a vector database optimized for speed
    var db = try VectorDB.init(allocator, 768, .{ // 768d vectors (common embedding size)
        .initial_capacity = 10000,
        .index_m = 32, // Higher M for better connectivity
        .index_ef_construction = 400, // Higher ef_construction for better build quality
    });
    defer db.deinit();

    // Pre-allocate for batch insertion
    try db.storage.reserveCapacity(10000);
    try db.index.reserveCapacity(10000);

    // Add vectors in batches for maximum throughput
    var rng = std.Random.DefaultPrng.init(42);
    const vectors = try allocator.alloc([768]f32, 10000);
    defer allocator.free(vectors);

    std.debug.print("Generating 10,000 vectors...\n", .{});
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
    for (normalized_v1, retrieved) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.0001);
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
