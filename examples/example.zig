const std = @import("std");
const VectorDB = @import("vector-db").VectorDB;

// Example: Building a simple image similarity search system
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Vector Database Example ===\n\n", .{});

    // 1. Create a database for 512-dimensional vectors (e.g., image embeddings)
    std.debug.print("Creating vector database...\n", .{});
    var db = try VectorDB.init(allocator, 512, .{
        .initial_capacity = 10000,
        .index_m = 32, // Higher M = better quality, more memory
        .index_ef_construction = 200, // Higher = better quality, slower indexing
        .file_path = "embeddings.db", // Persist to disk
    });
    defer db.deinit();

    // 2. Simulate adding image embeddings
    std.debug.print("Adding image embeddings...\n", .{});
    var rng = std.Random.DefaultPrng.init(42);

    // In real use, these would be actual image embeddings from a model
    var images_added: usize = 0;
    while (images_added < 1000) : (images_added += 1) {
        var embedding: [512]f32 = undefined;

        // Simulate different "categories" of images
        const category = images_added / 100;
        for (&embedding, 0..) |*v, i| {
            // Create clustered embeddings to simulate real data
            v.* = @as(f32, @floatFromInt(category)) * 0.1 +
                rng.random().float(f32) * 0.1 +
                @as(f32, @floatFromInt(i % 10)) * 0.01;
        }

        // Normalize the vector (common for embeddings)
        normalizeVector(&embedding);

        _ = try db.addVector(&embedding, null);

        if (images_added % 100 == 0) {
            std.debug.print("  Added {} embeddings\n", .{images_added});
        }
    }

    // 3. Search for similar images
    std.debug.print("\nSearching for similar images...\n", .{});

    // Create a query embedding (simulating a new image)
    var query_embedding: [512]f32 = undefined;
    const query_category: f32 = 3.0; // Should match category 3
    for (&query_embedding, 0..) |*v, i| {
        v.* = query_category * 0.1 +
            rng.random().float(f32) * 0.05 + // Less noise for clearer results
            @as(f32, @floatFromInt(i % 10)) * 0.01;
    }
    normalizeVector(&query_embedding);

    // Find top 10 most similar images
    const search_start = std.time.milliTimestamp();
    const results = try db.search(&query_embedding, 10);
    const search_time = std.time.milliTimestamp() - search_start;
    defer allocator.free(results);

    std.debug.print("Search completed in {} ms\n", .{search_time});
    std.debug.print("\nTop 10 similar images:\n", .{});
    for (results, 0..) |result, i| {
        const image_category = result.idx / 100;
        std.debug.print("  {}. Image {} (category {}) - similarity: {:.4}\n", .{
            i + 1,
            result.idx,
            image_category,
            1.0 - result.distance, // Convert distance to similarity
        });
    }

    // 4. Batch search example
    std.debug.print("\nPerforming batch search...\n", .{});
    var queries: [5][512]f32 = undefined;
    for (&queries, 0..) |*query, i| {
        for (query, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i)) * 0.1 +
                rng.random().float(f32) * 0.1 +
                @as(f32, @floatFromInt(j % 10)) * 0.01;
        }
        normalizeVector(query);
    }

    const query_slices = try allocator.alloc([]const f32, queries.len);
    defer allocator.free(query_slices);
    for (&queries, 0..) |*query, i| {
        query_slices[i] = query;
    }

    const batch_start = std.time.milliTimestamp();
    const batch_results = try db.searchBatch(query_slices, 5);
    const batch_time = std.time.milliTimestamp() - batch_start;
    defer {
        for (batch_results) |res| {
            allocator.free(res);
        }
        allocator.free(batch_results);
    }

    std.debug.print("Batch search for {} queries completed in {} ms\n", .{
        queries.len,
        batch_time,
    });
    std.debug.print("Average time per query: {:.2} ms\n", .{
        @as(f32, @floatFromInt(batch_time)) / @as(f32, @floatFromInt(queries.len)),
    });

    // 5. Save the database
    std.debug.print("\nSaving database to disk...\n", .{});
    try db.save("image_index.db");
    std.debug.print("Database saved successfully!\n", .{});
}

fn normalizeVector(vector: []f32) void {
    var sum: f32 = 0.0;
    for (vector) |v| {
        sum += v * v;
    }
    const norm = @sqrt(sum);
    if (norm > 0) {
        for (vector) |*v| {
            v.* /= norm;
        }
    }
}

// Example: Text embedding search
pub fn textSearchExample(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Text Search Example ===\n", .{});

    // Create a database for text embeddings (e.g., from BERT, 768 dimensions)
    var db = try VectorDB.init(allocator, 768, .{
        .initial_capacity = 50000,
        .index_m = 48,
        .index_ef_construction = 200,
    });
    defer db.deinit();

    // Simulate adding document embeddings
    const documents = [_][]const u8{
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world",
        "Zig is a general-purpose programming language",
        "Vector databases enable similarity search",
        "The weather today is sunny and warm",
    };

    std.debug.print("Adding document embeddings...\n", .{});
    for (documents, 0..) |doc, i| {
        var embedding: [768]f32 = undefined;
        // In real use, you'd use a model like BERT to generate embeddings
        generateMockTextEmbedding(doc, &embedding);
        _ = try db.addVector(&embedding, null);
        std.debug.print("  Added: \"{s}\"\n", .{doc});
        _ = i;
    }

    // Search for similar documents
    const query = "Programming languages and software development";
    var query_embedding: [768]f32 = undefined;
    generateMockTextEmbedding(query, &query_embedding);

    std.debug.print("\nSearching for documents similar to: \"{s}\"\n", .{query});
    const results = try db.search(&query_embedding, 3);
    defer allocator.free(results);

    std.debug.print("Most similar documents:\n", .{});
    for (results, 0..) |result, i| {
        if (result.idx < documents.len) {
            std.debug.print("  {}. \"{s}\" (similarity: {:.3})\n", .{
                i + 1,
                documents[result.idx],
                1.0 - result.distance,
            });
        }
    }
}

fn generateMockTextEmbedding(text: []const u8, embedding: []f32) void {
    // Simple mock embedding based on text characteristics
    // In practice, use a real embedding model
    var hash = std.hash.Wyhash.init(0);
    hash.update(text);
    const seed = hash.final();

    var rng = std.Random.DefaultPrng.init(seed);
    for (embedding) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0;
    }

    // Add some structure based on text length and common words
    const len_factor = @as(f32, @floatFromInt(text.len)) / 100.0;
    embedding[0] = len_factor;

    // Normalize
    normalizeVector(embedding);
}
