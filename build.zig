const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run unit tests");

    const main_mod = b.addModule("zonder", .{
        .root_source_file = .{ .path = "src/zonder.zig" },
    });

    const unit_tests_exe = b.addTest(.{
        .root_source_file = main_mod.root_source_file.?,
        .target = target,
        .optimize = optimize,
    });
    const unit_tests_run = b.addRunArtifact(unit_tests_exe);

    test_step.dependOn(&unit_tests_run.step);
}
