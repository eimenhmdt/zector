const std = @import("std");
const math = std.math;
const mem = std.mem;
const os = std.os;
const fs = std.fs;
const builtin = @import("builtin");

// Configuration constants
const VECTOR_ALIGNMENT = 32; // For AVX2
const PAGE_SIZE = 4096;
const INDEX_BUCKET_SIZE = 64;
const MMAP_THRESHOLD = 1024 * 1024; // 1MB
const PAGE_ALIGNMENT = 16384; // Common page size on macOS

// Vector operations using SIMD when available
const VectorOps = struct {
    // Compute dot product using SIMD instructions when available
    pub fn dot(a: []const f32, b: []const f32) f32 {
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
        var sum: f32 = 0.0;
        const vec_size = 8; // AVX2 can process 8 floats at once
        const simd_len = a.len - (a.len % vec_size);

        // Process SIMD chunks
        var i: usize = 0;
        while (i < simd_len) : (i += vec_size) {
            const va = @as(@Vector(8, f32), a[i..][0..8].*);
            const vb = @as(@Vector(8, f32), b[i..][0..8].*);
            const prod = va * vb;
            sum += @reduce(.Add, prod);
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    pub fn cosineDistance(a: []const f32, b: []const f32) f32 {
        const dot_product = dot(a, b);
        const norm_a = math.sqrt(dot(a, a));
        const norm_b = math.sqrt(dot(b, b));
        return 1.0 - (dot_product / (norm_a * norm_b));
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

    /// Cosine distance using pre-computed L2 norms of `a` and `b`.
    pub fn cosineDistanceWithNorms(a: []const f32, b: []const f32, norm_a: f32, norm_b: f32) f32 {
        const dot_product = dot(a, b);
        return 1.0 - (dot_product / (norm_a * norm_b));
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
    /// Pre-computed L2 norms (‖v‖) for every stored vector. Same length as `capacity`.
    norms: []f32,

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
            .norms = try allocator.alloc(f32, initial_capacity),
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
            .norms = try allocator.alloc(f32, 1024),
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
        self.allocator.free(self.norms);
        if (self.file) |file| {
            file.close();
        }
    }

    pub fn addVector(self: *VectorStorage, vector: []const f32) !usize {
        std.debug.assert(vector.len == self.dimension);

        if (self.count >= self.capacity) {
            try self.grow();
        }

        const idx = self.count;
        const offset = idx * self.dimension * @sizeOf(f32);
        const dest = self.data[offset..][0 .. vector.len * @sizeOf(f32)];
        @memcpy(dest, mem.sliceAsBytes(vector));

        // Compute and store L2 norm for cosine distance calculations
        var sum: f32 = 0.0;
        for (vector) |v| sum += v * v;
        self.norms[idx] = math.sqrt(sum);

        self.count += 1;
        return idx;
    }

    pub fn getVector(self: VectorStorage, idx: usize) []const f32 {
        std.debug.assert(idx < self.count);
        const offset = idx * self.dimension * @sizeOf(f32);
        const bytes = self.data[offset..][0 .. self.dimension * @sizeOf(f32)];
        return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..self.dimension];
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
            // Reallocate norms buffer as well
            self.norms = try self.allocator.realloc(self.norms, new_capacity);
        } else {
            self.data = try self.allocator.realloc(self.data, new_size);
            self.norms = try self.allocator.realloc(self.norms, new_capacity);
        }

        self.capacity = new_capacity;
    }

    /// Returns the pre-computed L2 norm for vector `idx`.
    pub fn getNorm(self: VectorStorage, idx: usize) f32 {
        return self.norms[idx];
    }
};

// Hierarchical Navigable Small World (HNSW) index for fast similarity search
const HNSWIndex = struct {
    const Connection = struct {
        offset: usize,
        len: usize,
        capacity: usize,
        worst_dist: f32, // Track the worst (largest) distance in this connection list
        worst_idx: usize, // Index within the connection list of the worst element
    };

    const Node = struct {
        level: u8,
        /// (offset, length) pairs for each level in the edge pool
        connections: []Connection,

        pub fn init(allocator: mem.Allocator, level: u8) !Node {
            const connections = try allocator.alloc(Connection, level + 1);
            for (connections) |*conn| {
                conn.* = .{ .offset = 0, .len = 0, .capacity = 0, .worst_dist = -1.0, .worst_idx = 0 };
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
    edge_pool_used: usize,
    /// Re-usable scratch vectors (grow-once arena)
    scratch_idx: []usize,
    scratch_dist: []f32,
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
            .edge_pool_used = 0,
            .scratch_idx = &[_]usize{},
            .scratch_dist = &[_]f32{},
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
        if (self.scratch_idx.len > 0) self.allocator.free(self.scratch_idx);
        if (self.scratch_dist.len > 0) self.allocator.free(self.scratch_dist);
    }

    /// Allocate space in the edge pool for `count` connections
    fn allocateEdges(self: *HNSWIndex, count: usize) !usize {
        const needed_capacity = self.edge_pool_used + count;
        if (needed_capacity > self.edge_pool.capacity) {
            const new_capacity = @max(needed_capacity, self.edge_pool.capacity * 2);
            try self.edge_pool.ensureTotalCapacity(new_capacity);
        }

        // Resize to accommodate new edges
        try self.edge_pool.resize(needed_capacity);

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

    /// Add a connection from node_idx to neighbor_idx at level
    fn addConnection(
        self: *HNSWIndex,
        node_idx: usize,
        neighbor_idx: usize,
        level: usize,
        dist: f32, // distance(node, neighbor)
        storage: *const VectorStorage, // for computing distances during pruning
    ) !void {
        const node = &self.nodes.items[node_idx];
        if (level > node.level) return;
        const cap = if (level == 0) self.m * 2 else self.m;

        var conn = &node.connections[level];

        // allocate on first use
        if (conn.capacity == 0) {
            conn.offset = try self.allocateEdges(cap);
            conn.capacity = cap;
            conn.len = 0;
            conn.worst_dist = -1.0;
            conn.worst_idx = 0;
        }

        // room left – just push
        if (conn.len < conn.capacity) {
            self.edge_pool.items[conn.offset + conn.len] = neighbor_idx;
            // Update worst tracking
            if (dist > conn.worst_dist) {
                conn.worst_dist = dist;
                conn.worst_idx = conn.len;
            }
            conn.len += 1;
            return;
        }

        // list full – replace worst if new is better
        if (dist < conn.worst_dist) {
            self.edge_pool.items[conn.offset + conn.worst_idx] = neighbor_idx;

            // Need to find new worst since we just replaced the old worst
            conn.worst_dist = -1.0;
            const vec = storage.getVector(node_idx);
            const norm = storage.getNorm(node_idx);
            for (0..conn.len) |i| {
                const id = self.edge_pool.items[conn.offset + i];
                const d = VectorOps.cosineDistanceWithNorms(
                    vec,
                    storage.getVector(id),
                    norm,
                    storage.getNorm(id),
                );
                if (d > conn.worst_dist) {
                    conn.worst_dist = d;
                    conn.worst_idx = i;
                }
            }
        }
    }

    fn selectLevel(self: *HNSWIndex) u8 {
        const f = self.rng.random().float(f64);
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
            return;
        }

        const vector = storage.getVector(idx);
        const vector_norm = storage.getNorm(idx); // Use pre-computed norm from storage
        var nearest = try self.searchLayer(vector, self.entry_point.?, 1, 0, storage);
        defer {
            for (nearest.items) |_| {}
            nearest.deinit();
        }

        // Search from top layer to target layer
        const entry_level = self.nodes.items[self.entry_point.?].level;
        if (entry_level > 0) {
            var l = @min(level, entry_level);
            while (l > 0) : (l -= 1) {
                const new_nearest = try self.searchLayer(vector, nearest.items[0], self.ef_construction, l, storage);
                nearest.deinit();
                nearest = new_nearest;
            }
        }

        // Search at level 0 with higher ef_construction for better quality
        const level0_nearest = try self.searchLayer(vector, nearest.items[0], self.ef_construction, 0, storage);
        nearest.deinit();
        nearest = level0_nearest;

        // Connect the new node
        var l: usize = 0;
        while (l <= level) : (l += 1) {
            const m = if (l == 0) self.m * 2 else self.m;
            const candidates = try self.topK(nearest.items, m, vector, vector_norm, storage);
            defer self.allocator.free(candidates);

            for (candidates) |neighbor| {
                const dist = VectorOps.cosineDistanceWithNorms(
                    vector,
                    storage.getVector(neighbor),
                    vector_norm,
                    storage.getNorm(neighbor),
                );
                try self.addConnection(idx, neighbor, l, dist, storage);
                // Only add reverse connection if neighbor has this level
                if (l <= self.nodes.items[neighbor].level) {
                    try self.addConnection(neighbor, idx, l, dist, storage); // symmetric distance
                }
            }
        }
    }

    fn searchLayer(
        self: *HNSWIndex,
        query: []const f32,
        entry: usize,
        num_closest: usize,
        layer: usize,
        storage: *const VectorStorage,
    ) !std.ArrayList(usize) {
        var visited = std.AutoHashMap(usize, void).init(self.allocator);
        defer visited.deinit();

        var candidates = std.PriorityQueue(SearchItem, void, compareSearchItems).init(self.allocator, {});
        defer candidates.deinit();

        // Use a priority queue for the dynamic candidate list W
        var w = std.PriorityQueue(SearchItem, void, compareSearchItemsReverse).init(self.allocator, {});
        defer w.deinit();

        const norm_query = math.sqrt(VectorOps.dot(query, query));
        const entry_dist = VectorOps.cosineDistanceWithNorms(query, storage.getVector(entry), norm_query, storage.getNorm(entry));
        try candidates.add(.{ .idx = entry, .distance = entry_dist });
        try w.add(.{ .idx = entry, .distance = entry_dist });
        try visited.put(entry, {});

        while (candidates.count() > 0) {
            const current = candidates.remove();
            if (w.count() > 0) {
                const worst_dist = w.peek().?.distance;
                if (current.distance > worst_dist) break;
            }

            // Skip if this node doesn't have connections at this layer
            if (layer > self.nodes.items[current.idx].level) continue;

            const connections = self.getConnections(current.idx, layer);
            for (connections) |neighbor| {
                if (!visited.contains(neighbor)) {
                    try visited.put(neighbor, {});
                    const dist = VectorOps.cosineDistanceWithNorms(query, storage.getVector(neighbor), norm_query, storage.getNorm(neighbor));

                    if (w.count() < num_closest) {
                        try candidates.add(.{ .idx = neighbor, .distance = dist });
                        try w.add(.{ .idx = neighbor, .distance = dist });
                    } else {
                        const worst_dist = w.peek().?.distance;
                        if (dist < worst_dist) {
                            try candidates.add(.{ .idx = neighbor, .distance = dist });
                            try w.add(.{ .idx = neighbor, .distance = dist });
                            // Remove the worst candidate
                            _ = w.remove();
                        }
                    }
                }
            }
        }

        // Convert priority queue back to ArrayList
        var result = std.ArrayList(usize).init(self.allocator);
        while (w.count() > 0) {
            const item = w.remove();
            try result.append(item.idx);
        }

        // Reverse the result since we want closest first
        std.mem.reverse(usize, result.items);
        return result;
    }

    /// Efficient top-k selection using scratch buffers and quickselect
    fn topK(
        self: *HNSWIndex,
        candidates: []usize,
        k: usize,
        base: []const f32,
        base_norm: f32,
        storage: *const VectorStorage,
    ) ![]usize {
        if (candidates.len <= k) {
            const result = try self.allocator.alloc(usize, candidates.len);
            @memcpy(result, candidates);
            return result;
        }

        // Make sure scratch is large enough – grow only when needed
        if (self.scratch_idx.len < candidates.len) {
            if (self.scratch_idx.len > 0) self.allocator.free(self.scratch_idx);
            if (self.scratch_dist.len > 0) self.allocator.free(self.scratch_dist);
            self.scratch_idx = try self.allocator.alloc(usize, candidates.len);
            self.scratch_dist = try self.allocator.alloc(f32, candidates.len);
        }

        // Fill scratch buffers
        for (candidates, 0..) |id, i| {
            self.scratch_idx[i] = i; // indices into candidates array
            self.scratch_dist[i] = VectorOps.cosineDistanceWithNorms(
                base,
                storage.getVector(id),
                base_norm,
                storage.getNorm(id),
            );
        }

        // Sort indices by distance and take the first k
        const context = self.scratch_dist[0..candidates.len];
        std.sort.heap(usize, self.scratch_idx[0..candidates.len], context, struct {
            fn lessThan(ctx: []f32, a: usize, b: usize) bool {
                return ctx[a] < ctx[b];
            }
        }.lessThan);

        // Return the k closest candidates
        const result = try self.allocator.alloc(usize, k);
        for (0..k) |i| {
            result[i] = candidates[self.scratch_idx[i]];
        }
        return result;
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
    allocator: mem.Allocator,
    storage: VectorStorage,
    index: HNSWIndex,
    metadata: std.StringHashMap(std.json.Value),
    search_ef: usize,

    pub fn init(allocator: mem.Allocator, dimension: usize, options: struct {
        initial_capacity: usize = 10000,
        index_m: usize = 16,
        index_ef_construction: usize = 200,
        file_path: ?[]const u8 = null,
    }) !VectorDB {
        const storage = if (options.file_path) |path|
            try VectorStorage.initWithFile(allocator, path, dimension)
        else
            try VectorStorage.init(allocator, dimension, options.initial_capacity);

        return VectorDB{
            .allocator = allocator,
            .storage = storage,
            .index = HNSWIndex.init(allocator, options.index_m, options.index_ef_construction),
            .metadata = std.StringHashMap(std.json.Value).init(allocator),
            .search_ef = 128, // Default search ef for good recall
        };
    }

    pub fn deinit(self: *VectorDB) void {
        self.storage.deinit();
        self.index.deinit();
        self.metadata.deinit();
    }

    pub fn addVector(self: *VectorDB, vector: []const f32, metadata: ?std.json.Value) !usize {
        const idx = try self.storage.addVector(vector);
        try self.index.insert(idx, &self.storage);

        if (metadata) |m| {
            const key = try std.fmt.allocPrint(self.allocator, "{}", .{idx});
            try self.metadata.put(key, m);
        }

        return idx;
    }

    pub fn search(self: *VectorDB, query: []const f32, k: usize) ![]SearchResult {
        if (self.index.entry_point == null) return &[_]SearchResult{};

        const ef = @max(k, self.search_ef);
        const candidates = try self.index.searchLayer(query, self.index.entry_point.?, ef, 0, &self.storage);
        defer candidates.deinit();

        const norm_query = math.sqrt(VectorOps.dot(query, query));
        var results = try self.allocator.alloc(SearchResult, @min(k, candidates.items.len));
        for (candidates.items[0..results.len], 0..) |idx, i| {
            results[i] = .{
                .idx = idx,
                .distance = VectorOps.cosineDistanceWithNorms(query, self.storage.getVector(idx), norm_query, self.storage.getNorm(idx)),
                .vector = self.storage.getVector(idx),
            };
        }

        // Sort by distance
        std.sort.heap(SearchResult, results, {}, struct {
            fn lessThan(context: void, a: SearchResult, b: SearchResult) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        return results;
    }

    pub const SearchResult = struct {
        idx: usize,
        distance: f32,
        vector: []const f32,
    };

    // Batch operations for efficiency
    pub fn addBatch(self: *VectorDB, vectors: []const []const f32) !void {
        for (vectors) |vector| {
            _ = try self.addVector(vector, null);
        }
    }

    pub fn searchBatch(self: *VectorDB, queries: []const []const f32, k: usize) ![][]SearchResult {
        var results = try self.allocator.alloc([]SearchResult, queries.len);
        for (queries, 0..) |query, i| {
            results[i] = try self.search(query, k);
        }
        return results;
    }

    // Persistence
    pub fn save(self: *VectorDB, path: []const u8) !void {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();

        // Write header
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

        // Read header
        const dimension = try file.reader().readInt(u32, .little);
        const count = try file.reader().readInt(u32, .little);

        // Initialize database
        var db = try VectorDB.init(allocator, dimension, .{
            .initial_capacity = count,
            .file_path = path,
        });

        // Load vectors
        db.storage.count = count;

        // Compute norms for the loaded vectors
        var idx: usize = 0;
        while (idx < count) : (idx += 1) {
            const vec = db.storage.getVector(idx);
            var sum: f32 = 0.0;
            for (vec) |v| sum += v * v;
            db.storage.norms[idx] = math.sqrt(sum);
        }

        // Load index structure
        // ... (implement index deserialization)

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

    // Create a vector database
    var db = try VectorDB.init(allocator, 128, .{
        .initial_capacity = 1000,
        .index_m = 16,
        .index_ef_construction = 200,
    });
    defer db.deinit();

    // Add some vectors
    var rng = std.Random.DefaultPrng.init(42);
    var vector: [128]f32 = undefined;

    std.debug.print("Adding vectors...\n", .{});
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        for (&vector) |*v| {
            v.* = rng.random().float(f32);
        }
        _ = try db.addVector(&vector, null);
    }

    // Search for similar vectors
    std.debug.print("Searching...\n", .{});
    for (&vector) |*v| {
        v.* = rng.random().float(f32);
    }

    const results = try db.search(&vector, 10);
    defer allocator.free(results);

    std.debug.print("Found {} results:\n", .{results.len});
    for (results) |result| {
        std.debug.print("  Index: {}, Distance: {:.4}\n", .{ result.idx, result.distance });
    }
}

test "vector operations" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const dot_product = VectorOps.dot(&a, &b);
    try std.testing.expectApproxEqAbs(dot_product, 20.0, 0.001);

    const cosine_dist = VectorOps.cosineDistance(&a, &b);
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
    try std.testing.expectEqualSlices(f32, &v1, retrieved);
}

test "cosine distance consistency" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    // Original calculation
    const old_dist = VectorOps.cosineDistance(&a, &b);

    // New calculation with precomputed norms
    const norm_a = math.sqrt(VectorOps.dot(&a, &a));
    const norm_b = math.sqrt(VectorOps.dot(&b, &b));
    const new_dist = VectorOps.cosineDistanceWithNorms(&a, &b, norm_a, norm_b);

    try std.testing.expectApproxEqAbs(old_dist, new_dist, 0.0001);

    // Test with random vectors
    var rng = std.Random.DefaultPrng.init(42);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var vec1: [128]f32 = undefined;
        var vec2: [128]f32 = undefined;
        for (&vec1) |*v| v.* = rng.random().float(f32);
        for (&vec2) |*v| v.* = rng.random().float(f32);

        const old_d = VectorOps.cosineDistance(&vec1, &vec2);
        const n1 = math.sqrt(VectorOps.dot(&vec1, &vec1));
        const n2 = math.sqrt(VectorOps.dot(&vec2, &vec2));
        const new_d = VectorOps.cosineDistanceWithNorms(&vec1, &vec2, n1, n2);

        try std.testing.expectApproxEqAbs(old_d, new_d, 0.0001);
    }
}
