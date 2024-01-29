const std = @import("std");
const assert = std.debug.assert;
const Ast = std.zig.Ast;

/// Helper function to get the root expression of the zon source.
pub fn rootExpr(ast: Ast) Ast.Node.Index {
    assert(ast.mode == .zon);
    return ast.nodes.items(.data)[0].lhs;
}

pub const NodeKind = enum {
    bool,
    char,
    enumeration,
    err,
    list,
    null,
    number,
    record,
    string,
    undefined,
    void,
};

pub const NodeKindError = error{
    UnrecognizedIdentifier,
    NonZonNodeTag,
    NonEmptyVoidBlock,
};
pub fn nodeKind(ast: Ast, node: Ast.Node.Index) NodeKindError!NodeKind {
    assert(node != 0);
    const nodes_tag: []const Ast.Node.Tag = ast.nodes.items(.tag);
    return switch (nodes_tag[node]) {
        .array_init_dot_two,
        .array_init_dot_two_comma,
        .array_init_dot,
        .array_init_dot_comma,
        .array_init,
        .array_init_comma,
        => .list,

        .struct_init_dot_two,
        .struct_init_dot_two_comma,
        .struct_init_dot,
        .struct_init_dot_comma,
        .struct_init,
        .struct_init_comma,
        => .record,

        .negation,
        .number_literal,
        => .number,

        .string_literal,
        .multiline_string_literal,
        => .string,

        .char_literal => .char,
        .enum_literal => .enumeration,
        .error_value => .err,

        .identifier => blk: {
            const nodes_main_token: []const Ast.TokenIndex = ast.nodes.items(.main_token);
            const main_token = nodes_main_token[node];
            const token_slice = ast.tokenSlice(main_token);
            break :blk std.ComptimeStringMap(NodeKind, .{
                .{ "false", .bool },
                .{ "null", .null },
                .{ "true", .bool },
                .{ "undefined", .undefined },
            }).get(token_slice) orelse error.UnrecognizedIdentifier;
        },

        .block => blk: {
            const nodes_data: []const Ast.Node.Data = ast.nodes.items(.data);
            const data = nodes_data[node];
            if (data.lhs != data.rhs) return error.NonEmptyVoidBlock;
            break :blk .void;
        },

        else => error.NonZonNodeTag,
    };
}

/// Assumes `nodeKind(ast, node) == .bool`.
pub fn boolValue(ast: Ast, node: Ast.Node.Index) bool {
    const main_token = ast.nodes.items(.main_token)[node];
    const token_slice = ast.tokenSlice(main_token);
    if (std.mem.eql(u8, "false", token_slice))
        return false
    else if (std.mem.eql(u8, "true", token_slice))
        return true;
    unreachable;
}

/// Assumes `nodeKind(ast, node) == .number`.
/// Also see `numberSrc`.
pub fn numberValue(ast: Ast, node: Ast.Node.Index, comptime T: type) (switch (@typeInfo(T)) {
    .Int => std.fmt.ParseIntError,
    .Float => std.fmt.ParseFloatError,
    else => error{},
})!T {
    const src = numberSrc(ast, node);
    return switch (@typeInfo(T)) {
        .Int => std.fmt.parseInt(T, src, 0),
        .Float => std.fmt.parseFloat(T, src),
        else => @compileError("Expected integer or floating point type, got " ++ @typeName(T)),
    };
}

/// Assumes `nodeKind(ast, node) == .number`.
pub fn numberSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    assert(nodeKind(ast, node) catch unreachable == .number);
    return ast.getNodeSource(node);
}

/// Assumes `nodeKind(ast, node) == .enumeration`.
/// Also see `enumerationSrc`.
pub fn enumerationValue(ast: Ast, node: Ast.Node.Index, comptime E: type) ?E {
    const src = enumerationSrc(ast, node);
    return std.meta.stringToEnum(E, src);
}

/// Assumes `nodeKind(ast, node) == .enumeration`.
pub fn enumerationSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    const nodes_main_token: []const Ast.TokenIndex = ast.nodes.items(.main_token);
    return ast.tokenSlice(nodes_main_token[node]);
}

pub const StringView = struct {
    first: Ast.TokenIndex,
    last: Ast.TokenIndex,

    /// Assumes `nodeKind(ast, node) == .string`.
    pub fn from(ast: Ast, node: Ast.Node.Index) StringView {
        assert(nodeKind(ast, node) catch unreachable == .string);
        const nodes_tag: []const Ast.Node.Tag = ast.nodes.items(.tag);
        switch (nodes_tag[node]) {
            .string_literal => {
                const nodes_main_token: []const Ast.TokenIndex = ast.nodes.items(.main_token);
                return .{
                    .first = nodes_main_token[node],
                    .last = nodes_main_token[node],
                };
            },
            .multiline_string_literal => {
                const nodes_data: []const Ast.Node.Data = ast.nodes.items(.data);
                const data = nodes_data[node];
                return .{
                    .first = data.lhs,
                    .last = data.rhs,
                };
            },
            else => unreachable,
        }
    }

    pub inline fn segmentCount(sv: StringView) u32 {
        return (sv.last - sv.first) + 1;
    }

    /// Assumes `index < sv.segmentCount()`.
    pub fn getSegment(
        sv: StringView,
        ast: Ast,
        index: Ast.TokenIndex,
    ) []const u8 {
        assert(index < sv.segmentCount());
        const token_index = sv.first + index;
        const token_slice = ast.tokenSlice(token_index);

        assert(token_slice.len >= 2);
        return switch (token_slice[0]) {
            '\\' => blk: {
                assert(token_slice[1] == '\\');
                const trim_newline =
                    token_index == sv.last and
                    token_slice.len > 2 and
                    token_slice[token_slice.len - 1] == '\n' //
                ;
                break :blk token_slice[2 .. token_slice.len - @intFromBool(trim_newline)];
            },
            '"' => blk: {
                assert(token_slice[token_slice.len - 1] == '"');
                break :blk token_slice[1 .. token_slice.len - 1];
            },
            else => unreachable,
        };
    }

    pub inline fn segmentIterator(sv: StringView, ast: Ast) SegmentIterator {
        return .{
            .ast = ast,
            .sv = sv,
            .i = 0,
        };
    }
    pub const SegmentIterator = struct {
        ast: Ast,
        sv: StringView,
        i: Ast.TokenIndex,

        pub inline fn next(iter: *SegmentIterator) ?[]const u8 {
            if (iter.i == iter.sv.segmentCount()) return null;
            defer iter.i += 1;
            return iter.sv.getSegment(iter.ast, iter.i);
        }
    };

    pub fn readerCtx(sv: StringView, ast: Ast) ReaderCtx {
        return .{
            .iter = sv.segmentIterator(ast),
            .current = "",
            .current_used = 0,
        };
    }

    pub const Reader = std.io.GenericReader(*ReaderCtx, error{}, ReaderCtx.read);
    pub const ReaderCtx = struct {
        iter: SegmentIterator,
        current: []const u8,
        current_used: usize,

        pub inline fn reader(ctx: *ReaderCtx) Reader {
            return .{ .context = ctx };
        }

        fn read(ctx: *ReaderCtx, bytes: []u8) error{}!usize {
            if (ctx.current_used == ctx.current.len) {
                ctx.current = ctx.iter.next() orelse return 0;
                ctx.current_used = 0;
            }
            const remaining = ctx.current[ctx.current_used..];
            const amt = @min(remaining.len, bytes.len);
            @memcpy(bytes[0..amt], remaining[0..amt]);
            ctx.current_used += amt;
            return amt;
        }
    };
};

test "null" {
    const allocator = std.testing.allocator;

    var ast = try Ast.parse(allocator, "null", .zon);
    defer ast.deinit(allocator);

    const root_expr = rootExpr(ast);
    try std.testing.expectEqual(.null, nodeKind(ast, root_expr));
}

test "undefined" {
    const allocator = std.testing.allocator;

    var ast = try Ast.parse(allocator, "undefined", .zon);
    defer ast.deinit(allocator);

    const root_expr = rootExpr(ast);
    try std.testing.expectEqual(.undefined, nodeKind(ast, root_expr));
}

test "bool" {
    const allocator = std.testing.allocator;

    {
        var ast = try Ast.parse(allocator, "true", .zon);
        defer ast.deinit(allocator);

        const root_expr = rootExpr(ast);
        try std.testing.expectEqual(.bool, nodeKind(ast, root_expr));
        try std.testing.expectEqual(true, boolValue(ast, root_expr));
    }
    {
        var ast = try Ast.parse(allocator, "false", .zon);
        defer ast.deinit(allocator);

        const root_expr = rootExpr(ast);
        try std.testing.expectEqual(.bool, nodeKind(ast, root_expr));
        try std.testing.expectEqual(false, boolValue(ast, root_expr));
    }
}

test "number" {
    const allocator = std.testing.allocator;

    {
        var ast = try Ast.parse(allocator, "-0341_55", .zon);
        defer ast.deinit(allocator);

        const root_expr = rootExpr(ast);
        try std.testing.expectEqual(.number, nodeKind(ast, root_expr));
        try std.testing.expectEqualStrings("-0341_55", numberSrc(ast, root_expr));
    }
}

test "enumeration" {
    const allocator = std.testing.allocator;

    var ast = try Ast.parse(allocator, ".foo", .zon);
    defer ast.deinit(allocator);

    const root_expr = rootExpr(ast);
    try std.testing.expectEqual(.enumeration, nodeKind(ast, root_expr));
    try std.testing.expectEqualStrings("foo", enumerationSrc(ast, root_expr));
}

test StringView {
    // just to demo the in-source equivalent of multiline string literal newlines:
    std.testing.expectEqualStrings("foo\nbar",
        \\foo
        \\bar
        // newline is here, but not included in mutliline string
    ) catch unreachable;
    std.testing.expectEqualStrings("foo\nbar\n",
        \\foo
        \\bar
        \\
        // previous newline is included in the multiline string
    ) catch unreachable;

    try testStringViewSrc(
        \\ \\foo
        \\ \\bar
        \\
    , &.{ "foo\n", "bar" });

    try testStringViewSrc(
        \\ \\foo
        \\ \\bar
        \\ \\
        \\
    , &.{ "foo\n", "bar\n", "" });

    try testStringViewSrc(
        \\"foo bar"
    , &.{"foo bar"});

    try testStringViewSrc(
        \\"foo\nbar"
    , &.{"foo\\nbar"});
}

fn testStringViewSrc(
    src: [:0]const u8,
    expected_segments: []const []const u8,
) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testStringViewAst(ast, rootExpr(ast), expected_segments);
}
fn testStringViewAst(
    ast: Ast,
    node: Ast.Node.Index,
    expected_segments: []const []const u8,
) !void {
    try std.testing.expectEqual(.string, nodeKind(ast, node));

    const sv = StringView.from(ast, node);

    var iter = sv.segmentIterator(ast);

    var actual_segments = std.ArrayList([]const u8).init(std.testing.allocator);
    defer actual_segments.deinit();

    while (iter.next()) |actual_segment| try actual_segments.append(actual_segment);
    try std.testing.expectEqualDeep(expected_segments, actual_segments.items);

    // more of a sanity check than anything, makes it so we're covering all public APIs.
    for (actual_segments.items, 0..) |actual_segment, i| {
        errdefer testPrint("Difference occurred on segment {d}\n", .{i});
        try std.testing.expectEqualStrings(sv.getSegment(ast, @intCast(i)), actual_segment);
    }

    var expected_str = std.ArrayList(u8).init(std.testing.allocator);
    defer expected_str.deinit();
    for (expected_segments) |segment| try expected_str.appendSlice(segment);

    var actual_str = std.ArrayList(u8).init(std.testing.allocator);
    defer actual_str.deinit();
    {
        var read_ctx = sv.readerCtx(ast);
        const reader = read_ctx.reader();

        const Fifo = std.fifo.LinearFifo(u8, .{ .Static = 4096 });
        var fifo: Fifo = Fifo.init();
        try fifo.pump(reader, actual_str.writer());
    }

    try std.testing.expectEqualStrings(expected_str.items, actual_str.items);
}

fn testPrint(comptime fmt: []const u8, args: anytype) void {
    if (@inComptime()) {
        @compileError(std.fmt.comptimePrint(fmt, args));
    } else if (std.testing.backend_can_print) {
        std.debug.print(fmt, args);
    }
}
