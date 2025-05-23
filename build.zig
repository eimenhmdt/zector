// const std = @import("std");

// pub fn build(b: *std.Build) void {
//     const target = b.standardTargetOptions(.{});
//     const optimize = b.standardOptimizeOption(.{});

//     // Library
//     const lib = b.addStaticLibrary(.{
//         .name = "vector-db",
//         .root_source_file = b.path("src/main.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     b.installArtifact(lib);

//     // Main executable
//     const exe = b.addExecutable(.{
//         .name = "vector-db-demo",
//         .root_source_file = b.path("src/main.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     b.installArtifact(exe);

//     // Run command
//     const run_cmd = b.addRunArtifact(exe);
//     run_cmd.step.dependOn(b.getInstallStep());
//     if (b.args) |args| {
//         run_cmd.addArgs(args);
//     }
//     const run_step = b.step("run", "Run the demo");
//     run_step.dependOn(&run_cmd.step);

//     // Example executable
//     const example = b.addExecutable(.{
//         .name = "example",
//         .root_source_file = b.path("examples/example.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     example.root_module.addImport("vector-db", lib.root_module);

//     const run_example = b.addRunArtifact(example);
//     const example_step = b.step("example", "Run the example");
//     example_step.dependOn(&run_example.step);

//     // OpenAI example
//     const openai_example = b.addExecutable(.{
//         .name = "openai-example",
//         .root_source_file = b.path("examples/openai_example.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     openai_example.root_module.addImport("vector-db", lib.root_module);

//     const run_openai = b.addRunArtifact(openai_example);
//     const openai_step = b.step("openai", "Run the OpenAI embedding example");
//     openai_step.dependOn(&run_openai.step);

//     // Tests
//     const lib_unit_tests = b.addTest(.{
//         .root_source_file = b.path("src/main.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
//     const test_step = b.step("test", "Run unit tests");
//     test_step.dependOn(&run_lib_unit_tests.step);
// }

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library
    const lib = b.addStaticLibrary(.{
        .name = "vector-db",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Main executable
    const exe = b.addExecutable(.{
        .name = "vector-db-demo",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the demo");
    run_step.dependOn(&run_cmd.step);

    // Example executable
    const example = b.addExecutable(.{
        .name = "example",
        .root_source_file = b.path("examples/example.zig"),
        .target = target,
        .optimize = optimize,
    });
    example.root_module.addImport("vector-db", lib.root_module);

    const run_example = b.addRunArtifact(example);
    const example_step = b.step("example", "Run the example");
    example_step.dependOn(&run_example.step);

    // OpenAI example
    const openai_example = b.addExecutable(.{
        .name = "openai-example",
        .root_source_file = b.path("examples/openai_example.zig"),
        .target = target,
        .optimize = optimize,
    });
    openai_example.root_module.addImport("vector-db", lib.root_module);

    const run_openai = b.addRunArtifact(openai_example);
    const openai_step = b.step("openai", "Run the OpenAI embedding example");
    openai_step.dependOn(&run_openai.step);

    // Benchmark
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("examples/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast, // Important for benchmarks!
    });
    benchmark.root_module.addImport("vector-db", lib.root_module);

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run performance benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);

    // Tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
