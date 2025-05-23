const std = @import("std");
const VectorDB = @import("vector-db").VectorDB;
const builtin = @import("builtin");

const c = @cImport({
    @cInclude("sys/resource.h");
});

const BenchmarkResults = struct {
    name: []const u8,
    vectors: usize,
    dimension: usize,
    build_time_ms: f64,
    vectors_per_sec: f64,
    search_time_us: f64,
    qps: f64, // queries per second
    recall_at_10: f64,
    memory_mb: f64,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Vector Database Performance Benchmark ===\n\n", .{});

    // Test configurations
    const configs = [_]struct {
        vectors: usize,
        dim: usize,
        name: []const u8,
    }{
        .{ .vectors = 1_000, .dim = 128, .name = "Small (1K vectors, 128d)" },
        .{ .vectors = 10_000, .dim = 384, .name = "Medium (10K vectors, 384d)" },
        .{ .vectors = 50_000, .dim = 768, .name = "Large (50K vectors, 768d)" },
    };

    var results = std.ArrayList(BenchmarkResults).init(allocator);
    defer results.deinit();

    for (configs) |config| {
        std.debug.print("Running benchmark: {s}\n", .{config.name});
        const result = try runBenchmark(allocator, config.vectors, config.dim, config.name);
        try results.append(result);
        std.debug.print("\n", .{});
    }

    // Print comparison table
    printComparisonTable(results.items);
}

fn runBenchmark(allocator: std.mem.Allocator, num_vectors: usize, dimension: usize, name: []const u8) !BenchmarkResults {
    const start_memory = try getMemoryUsage();

    // Create database
    var db = try VectorDB.init(allocator, dimension, .{
        .initial_capacity = num_vectors,
        .index_m = 48, // Increased for better quality
        .index_ef_construction = 600, // Higher for better recall
    });
    defer db.deinit();

    // Increase search ef for higher recall (≥ 10 × k)
    db.setSearchEf(512);

    // Generate random vectors
    var rng = std.Random.DefaultPrng.init(42);
    const vectors = try allocator.alloc([]f32, num_vectors);
    defer {
        for (vectors) |v| allocator.free(v);
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dimension);
        for (vec.*) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }
        normalizeVector(vec.*);
    }

    // Benchmark insertion
    std.debug.print("  Building index...\n", .{});
    const build_start = std.time.milliTimestamp();

    // Use optimized batch insertion
    try db.addBatch(vectors);

    const build_time = @as(f64, @floatFromInt(std.time.milliTimestamp() - build_start));
    const vectors_per_sec = @as(f64, @floatFromInt(num_vectors)) * 1000.0 / build_time;

    std.debug.print("  ✓ Indexed {} vectors in {:.1} ms ({:.0} vectors/sec)\n", .{
        num_vectors,
        build_time,
        vectors_per_sec,
    });

    // Benchmark search
    const num_queries = 1000;
    var query_times = try allocator.alloc(i64, num_queries);
    defer allocator.free(query_times);

    std.debug.print("  Running {} search queries...\n", .{num_queries});

    // Generate separate query vectors for proper recall measurement
    const query_vectors = try allocator.alloc([]f32, num_queries);
    defer {
        for (query_vectors) |v| allocator.free(v);
        allocator.free(query_vectors);
    }

    for (query_vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dimension);
        for (vec.*) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }
        normalizeVector(vec.*);
    }

    var total_recall: f64 = 0.0;
    for (0..num_queries) |i| {
        const query = query_vectors[i];

        const search_start = std.time.microTimestamp();
        const results = try db.search(query, 10);
        query_times[i] = std.time.microTimestamp() - search_start;

        // Calculate ground truth by brute force - use full dataset for ≤ 10K vectors
        const sample_size = if (vectors.len <= 10_000) vectors.len else @min(1000, vectors.len);
        var ground_truth = try allocator.alloc(struct { idx: usize, distance: f32 }, sample_size);
        defer allocator.free(ground_truth);

        for (0..sample_size) |j| {
            const vec_idx = if (vectors.len <= 10_000) j else (i * 17 + j) % vectors.len; // full or pseudo-random sampling
            var dot_sum: f32 = 0.0;
            for (query, vectors[vec_idx]) |q, v| {
                dot_sum += q * v;
            }
            ground_truth[j] = .{ .idx = vec_idx, .distance = 1.0 - dot_sum }; // cosine distance
        }

        // Sort ground truth by distance
        std.sort.heap(@TypeOf(ground_truth[0]), ground_truth, {}, struct {
            fn lessThan(context: void, a: @TypeOf(ground_truth[0]), b: @TypeOf(ground_truth[0])) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        // Calculate recall@10
        const k = @min(10, ground_truth.len);
        var recall_count: usize = 0;
        for (results) |result| {
            for (ground_truth[0..k]) |gt| {
                if (result.idx == gt.idx) {
                    recall_count += 1;
                    break;
                }
            }
        }
        total_recall += @as(f64, @floatFromInt(recall_count)) / @as(f64, @floatFromInt(k));

        allocator.free(results);
    }

    // Calculate statistics
    var sum: i64 = 0;
    var min: i64 = query_times[0];
    var max: i64 = query_times[0];
    for (query_times) |time| {
        sum += time;
        min = @min(min, time);
        max = @max(max, time);
    }
    const avg_time = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(num_queries));
    const qps = 1_000_000.0 / avg_time;
    const recall = total_recall / @as(f64, @floatFromInt(num_queries));

    std.debug.print("  ✓ Search performance:\n", .{});
    std.debug.print("    - Average: {:.1} μs/query ({:.0} QPS)\n", .{ avg_time, qps });
    std.debug.print("    - Min/Max: {} μs / {} μs\n", .{ min, max });
    std.debug.print("    - Recall@10: {:.1}%\n", .{recall * 100.0});

    const end_memory = try getMemoryUsage();
    const memory_mb = @as(f64, @floatFromInt(end_memory - start_memory)) / 1024.0 / 1024.0;

    return BenchmarkResults{
        .name = try allocator.dupe(u8, name),
        .vectors = num_vectors,
        .dimension = dimension,
        .build_time_ms = build_time,
        .vectors_per_sec = vectors_per_sec,
        .search_time_us = avg_time,
        .qps = qps,
        .recall_at_10 = recall * 100.0,
        .memory_mb = memory_mb,
    };
}

fn normalizeVector(vec: []f32) void {
    var sum: f32 = 0.0;
    for (vec) |v| sum += v * v;
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (vec) |*v| v.* /= norm;
    }
}

fn getMemoryUsage() !usize {
    comptime {
        if (builtin.os.tag != .macos and builtin.os.tag != .linux) {
            @compileError("Memory usage tracking only supported on macOS and Linux");
        }
    }

    var usage: c.struct_rusage = undefined;
    if (c.getrusage(c.RUSAGE_SELF, &usage) != 0) return error.GetrusageFailed;

    // On macOS ru_maxrss is in kilobytes, on Linux it's in kilobytes too on modern systems
    // Convert to bytes for consistency
    return @as(usize, @intCast(usage.ru_maxrss)) * 1024;
}

fn printComparisonTable(results: []const BenchmarkResults) void {
    std.debug.print("\n=== Performance Summary ===\n\n", .{});
    std.debug.print("{s:<30} | {s:>10} | {s:>15} | {s:>12} | {s:>10} | {s:>11}\n", .{
        "Configuration",
        "Index Time",
        "Throughput",
        "Search Time",
        "QPS",
        "Recall@10",
    });
    std.debug.print("{s:-<30} | {s:->10} | {s:->15} | {s:->12} | {s:->10} | {s:->11}\n", .{
        "",
        "",
        "",
        "",
        "",
        "",
    });

    for (results) |r| {
        std.debug.print("{s:<30} | {d:>8.1} ms | {d:>13.0} v/s | {d:>10.1} μs | {d:>10.0} | {d:>9.1}%\n", .{
            r.name,
            r.build_time_ms,
            r.vectors_per_sec,
            r.search_time_us,
            r.qps,
            r.recall_at_10,
        });
    }
}
