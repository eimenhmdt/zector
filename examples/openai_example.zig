const std = @import("std");
const VectorDB = @import("vector-db").VectorDB;

// OpenAI API configuration
const OPENAI_API_URL = "https://api.openai.com/v1/embeddings";
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIM = 1536; // Default dimension for text-embedding-3-small

// OpenAI API structures
const EmbeddingRequest = struct {
    model: []const u8,
    input: []const u8,
    encoding_format: []const u8 = "float",
};

const EmbeddingResponse = struct {
    object: []const u8,
    data: []struct {
        object: []const u8,
        index: usize,
        embedding: []f32,
    },
    model: []const u8,
    usage: struct {
        prompt_tokens: u32,
        total_tokens: u32,
    },
};

// Get embedding from OpenAI API
pub fn getEmbedding(allocator: std.mem.Allocator, text: []const u8, api_key: []const u8) ![]f32 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    // Prepare request body
    const request_body = try std.json.stringifyAlloc(allocator, EmbeddingRequest{
        .model = EMBEDDING_MODEL,
        .input = text,
        .encoding_format = "float",
    }, .{});
    defer allocator.free(request_body);

    // Prepare headers
    const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
    defer allocator.free(auth_header);

    const headers = &[_]std.http.Header{
        .{ .name = "Authorization", .value = auth_header },
        .{ .name = "Content-Type", .value = "application/json" },
    };

    // Create request
    const uri = try std.Uri.parse(OPENAI_API_URL);

    var response_body = std.ArrayList(u8).init(allocator);
    defer response_body.deinit();

    const response = try client.fetch(.{
        .method = .POST,
        .location = .{ .uri = uri },
        .extra_headers = headers,
        .payload = request_body,
        .response_storage = .{ .dynamic = &response_body },
    });

    if (response.status != .ok) {
        std.debug.print("Error: HTTP {}\n", .{response.status});
        return error.ApiError;
    }

    // Response body is now in response_body.items
    const body = response_body.items;

    // Parse JSON response
    const parsed = try std.json.parseFromSlice(EmbeddingResponse, allocator, body, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    // Extract embedding
    if (parsed.value.data.len == 0) {
        return error.NoEmbeddingReturned;
    }

    // Copy embedding to return
    const embedding = try allocator.alloc(f32, parsed.value.data[0].embedding.len);
    @memcpy(embedding, parsed.value.data[0].embedding);

    return embedding;
}

// Main example using real OpenAI embeddings
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get API key from environment
    const api_key = std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY") catch {
        std.debug.print("Error: OPENAI_API_KEY environment variable not set\n", .{});
        std.debug.print("Set it with: export OPENAI_API_KEY='your-api-key'\n", .{});
        return;
    };
    defer allocator.free(api_key);

    std.debug.print("=== Vector Database with OpenAI Embeddings ===\n\n", .{});

    // Create vector database
    var db = try VectorDB.init(allocator, EMBEDDING_DIM, .{
        .initial_capacity = 1000,
        .index_m = 32,
        .index_ef_construction = 200,
        .file_path = "openai_embeddings.db",
    });
    defer db.deinit();

    // Sample documents to embed
    const documents = [_][]const u8{
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typography.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Zig is a general-purpose programming language designed for robustness, optimality, and maintainability.",
        "Vector databases enable similarity search by storing and indexing high-dimensional embeddings.",
        "The weather today is sunny with a high of 75 degrees Fahrenheit and clear skies.",
        "OpenAI's GPT models are large language models trained on diverse internet text.",
        "Rust and Zig are both systems programming languages focused on memory safety and performance.",
        "Semantic search understands the meaning and context of queries rather than just keyword matching.",
        "The recipe calls for two cups of flour, one cup of sugar, and three eggs.",
        "Database indexing improves query performance by creating data structures for faster lookups.",
    };

    // Add documents to the database
    std.debug.print("Adding documents with OpenAI embeddings...\n", .{});
    for (documents, 0..) |doc, i| {
        std.debug.print("  [{}/{}] Embedding: {s}\n", .{ i + 1, documents.len, doc[0..@min(50, doc.len)] });

        const embedding = try getEmbedding(allocator, doc, api_key);
        defer allocator.free(embedding);

        _ = try db.addVector(embedding, null);
    }

    // Perform semantic search
    std.debug.print("\n=== Semantic Search Examples ===\n", .{});

    const queries = [_][]const u8{
        "programming languages for systems development",
        "artificial intelligence and data",
        "how to make a cake",
        "storing vectors for similarity matching",
        "what's the weather like",
    };

    for (queries) |query| {
        std.debug.print("\nQuery: \"{s}\"\n", .{query});

        const query_embedding = try getEmbedding(allocator, query, api_key);
        defer allocator.free(query_embedding);

        const results = try db.search(query_embedding, 3);
        defer allocator.free(results);

        std.debug.print("Top {} results:\n", .{results.len});
        for (results, 0..) |result, i| {
            const similarity = 1.0 - result.distance;
            std.debug.print("  {}. Document {}: {s}... (similarity: {:.3})\n", .{
                i + 1,
                result.idx,
                documents[result.idx][0..@min(60, documents[result.idx].len)],
                similarity,
            });
        }
    }

    // Save the database
    std.debug.print("\nSaving database...\n", .{});
    try db.save("openai_vectors.db");
    std.debug.print("Database saved successfully!\n", .{});
}
