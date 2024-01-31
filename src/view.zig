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
    empty_tuple,
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
    TypedArrayInit,
    TypedStructInit,
};
pub fn nodeKind(ast: Ast, node: Ast.Node.Index) NodeKindError!NodeKind {
    assert(node != 0);
    const nodes_tag: []const Ast.Node.Tag = ast.nodes.items(.tag);
    const nodes_data: []const Ast.Node.Data = ast.nodes.items(.data);
    return switch (nodes_tag[node]) {
        .array_init_dot_two,
        .array_init_dot_two_comma,
        .array_init_dot,
        .array_init_dot_comma,
        => .list,
        .array_init,
        .array_init_comma,
        => blk: {
            const data = nodes_data[node];
            assert(data.lhs != 0); // the doc comment says it can be ommitted, however this never occurs in practice at the time of writing.
            break :blk error.TypedArrayInit;
        },

        .struct_init_dot_two => blk: {
            const data = nodes_data[node];
            if (data.lhs == 0) {
                assert(data.rhs == 0);
                break :blk .empty_tuple;
            }
            break :blk .record;
        },
        .struct_init_dot_two_comma,
        .struct_init_dot,
        .struct_init_dot_comma,
        => .record,
        .struct_init,
        .struct_init_comma,
        => blk: {
            const data = nodes_data[node];
            assert(data.lhs != 0); // the doc comment says it can be ommitted, however this never occurs in practice at the time of writing.
            break :blk error.TypedStructInit;
        },

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

        .block_two => blk: {
            const data = nodes_data[node];
            if (data.lhs != 0) return error.NonEmptyVoidBlock;
            if (data.rhs != 0) return error.NonEmptyVoidBlock;
            break :blk .void;
        },

        else => error.NonZonNodeTag,
    };
}

fn testNodeKindSrc(expected: NodeKindError!NodeKind, src: [:0]const u8) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testNodeKindAst(expected, ast, rootExpr(ast));
}

fn testNodeKindAst(expected: NodeKindError!NodeKind, ast: Ast, node: Ast.Node.Index) !void {
    const actual = nodeKind(ast, node);
    try std.testing.expectEqual(expected, actual);
}

test "null, undefined, void, & empty tuple" {
    try testNodeKindSrc(.null, "null");
    try testNodeKindSrc(.undefined, "undefined");
    try testNodeKindSrc(.void, "{}");
    try testNodeKindSrc(.empty_tuple, ".{}");
}

/// Assumes `nodeKind(ast, node) == .bool`.
pub fn boolValue(ast: Ast, node: Ast.Node.Index) bool {
    const src = boolSrc(ast, node);
    if (std.mem.eql(u8, "false", src)) return false;
    if (std.mem.eql(u8, "true", src)) return true;
    unreachable;
}

/// Assumes `nodeKind(ast, node) == .bool`.
pub inline fn boolSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    assert(nodeKind(ast, node) catch unreachable == .bool);
    return ast.tokenSlice(ast.nodes.items(.main_token)[node]);
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

/// Assumes `nodeKind(ast, node) == .number`.
pub fn numberSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    assert(nodeKind(ast, node) catch unreachable == .number);
    return ast.getNodeSource(node);
}

fn testNumberSrc(
    src: [:0]const u8,
) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testNumberAst(src, ast, rootExpr(ast));
}
fn testNumberAst(expected_value: []const u8, ast: Ast, node: Ast.Node.Index) !void {
    try testNodeKindAst(.number, ast, node);
    try std.testing.expectEqualStrings(expected_value, numberSrc(ast, node));
}

test "number" {
    try testNumberSrc("-34_155");
    try testNumberSrc("34_155");
    try testNumberSrc("4_2_1");
    try testNumberSrc("2.3");
    try testNumberSrc("2.3e7");
    try testNumberSrc("2.3e-8");
    try testNumberSrc("2.3e-122");
}

/// Assumes `nodeKind(ast, node) == .err`.
pub inline fn errValue(ast: Ast, node: Ast.Node.Index, comptime E: type) ?E {
    const src = errSrc(ast, node);
    inline for (@typeInfo(E).ErrorSet orelse @compileError("Expected concrete error set, got " ++ @typeName(E))) |err| {
        if (std.mem.eql(u8, src, err.name)) return @field(E, err.name);
    }
    return null;
}

/// Assumes `nodeKind(ast, node) == .err`.
pub inline fn errSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    assert(nodeKind(ast, node) catch unreachable == .err);
    const data: Ast.Node.Data = ast.nodes.items(.data)[node];
    return ast.tokenSlice(data.rhs);
}

fn testErrSrc(expected: anytype, src: [:0]const u8) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testErrAst(expected, ast, rootExpr(ast));
}
fn testErrAst(expected: anytype, ast: Ast, node: Ast.Node.Index) !void {
    try testNodeKindAst(.err, ast, node);
    try std.testing.expectEqualStrings(@errorName(expected), errSrc(ast, node));
    try std.testing.expectEqual(@as(?@TypeOf(expected), expected), errValue(ast, node, @TypeOf(expected)));
}

test "err" {
    try testErrSrc(error.Foo, "error.Foo");
    try testErrSrc(error{ Foo, Bar }.Bar, "error.Bar");
}

/// Assumes `nodeKind(ast, node) == .enumeration`.
/// Also see `enumerationSrc`.
pub inline fn enumerationValue(ast: Ast, node: Ast.Node.Index, comptime E: type) ?E {
    return std.meta.stringToEnum(E, enumerationSrc(ast, node));
}

/// Assumes `nodeKind(ast, node) == .enumeration`.
pub fn enumerationSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    assert(nodeKind(ast, node) catch unreachable == .enumeration);
    const nodes_main_token: []const Ast.TokenIndex = ast.nodes.items(.main_token);
    return ast.tokenSlice(nodes_main_token[node]);
}

fn testEnumerationSrc(expected_value: anytype, src: [:0]const u8) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testEnumerationAst(expected_value, ast, rootExpr(ast));
}
fn testEnumerationAst(expected_value: anytype, ast: Ast, node: Ast.Node.Index) !void {
    try testNodeKindAst(.enumeration, ast, node);
    const expected_str: []const u8, //
    const already_a_string: bool //
    = switch (@typeInfo(@TypeOf(expected_value))) {
        .Enum, .EnumLiteral => .{ @tagName(expected_value), false },
        .Pointer => .{ expected_value, true },
        else => @compileError("Expected enum value, enum literal, or string, got " ++ @typeName(@TypeOf(expected_value))),
    };
    try std.testing.expectEqualStrings(expected_str, enumerationSrc(ast, node));
    if (comptime !already_a_string) {
        const ExpectedType = switch (@typeInfo(@TypeOf(expected_value))) {
            .Enum => @TypeOf(expected_value),
            .EnumLiteral => @Type(.{
                .Enum = .{
                    // for some reason using u0 here causes a compile error in `stringToEnum`.
                    .tag_type = u1,
                    .is_exhaustive = true,
                    .fields = &.{
                        .{ .name = @tagName(expected_value), .value = 0 },
                    },
                    .decls = &.{},
                },
            }),
            else => unreachable,
        };
        try std.testing.expectEqual(expected_value, enumerationValue(ast, node, ExpectedType));
    }
}

test "enumeration" {
    try testEnumerationSrc(.foo, ".foo");
    try testEnumerationSrc(enum { fizz, buzz, bar }.bar, ".bar");
    try testEnumerationSrc("baz", ".baz");
}

pub const CharValueError = error{InvalidCharacterLiteral};
/// Assumes `nodeKind(ast, node) == .char`.
/// Also see `charSrc` & `charSrcQuoted`.
/// Returns the unicode point value represented by the character literal.
pub fn charValue(ast: Ast, node: Ast.Node.Index) CharValueError!u21 {
    const quoted_src = charSrcQuoted(ast, node);
    const parsed_char_literal = std.zig.parseCharLiteral(quoted_src);
    return switch (parsed_char_literal) {
        .success => |codepoint| codepoint,
        .failure => error.InvalidCharacterLiteral,
    };
}

/// Assumes `nodeKind(ast, node) == .char`.
pub fn charSrc(ast: Ast, node: Ast.Node.Index) []const u8 {
    const quoted_src = charSrcQuoted(ast, node);
    return quoted_src[1 .. quoted_src.len - 1];
}

/// Assumes `nodeKind(ast, node) == .char`.
/// Returned string includes the surrounding single quotes.
pub fn charSrcQuoted(ast: Ast, node: Ast.Node.Index) []const u8 {
    assert(nodeKind(ast, node) catch unreachable == .char);
    const nodes_main_token: []const Ast.TokenIndex = ast.nodes.items(.main_token);
    const token_slice = ast.tokenSlice(nodes_main_token[node]);
    assert(token_slice.len >= 2);
    assert('\'' == token_slice[0]);
    assert('\'' == token_slice[token_slice.len - 1]);
    return token_slice;
}

fn testCharSrc(expected_value: u21, src: [:0]const u8) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testCharAst(expected_value, ast, rootExpr(ast));
}
fn testCharAst(expected_value: u21, ast: Ast, node: Ast.Node.Index) !void {
    try testNodeKindAst(.char, ast, node);
    try std.testing.expectEqual(expected_value, charValue(ast, node));
}

test "char" {
    try testCharSrc('a', "'a'");
    try testCharSrc('\u{21}', "'\\u{21}'");
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

fn testStringViewSrc(
    expected_segments: []const []const u8,
    src: [:0]const u8,
) !void {
    var ast = try Ast.parse(std.testing.allocator, src, .zon);
    defer ast.deinit(std.testing.allocator);
    return testStringViewAst(expected_segments, ast, rootExpr(ast));
}
fn testStringViewAst(
    expected_segments: []const []const u8,
    ast: Ast,
    node: Ast.Node.Index,
) !void {
    try testNodeKindAst(.string, ast, node);
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

    try testStringViewSrc(&.{ "foo\n", "bar" },
        \\ \\foo
        \\ \\bar
        \\
    );

    try testStringViewSrc(&.{ "foo\n", "bar\n", "" },
        \\ \\foo
        \\ \\bar
        \\ \\
        \\
    );

    try testStringViewSrc(&.{"foo bar"},
        \\"foo bar"
    );

    try testStringViewSrc(&.{"foo\\nbar"},
        \\"foo\nbar"
    );
}

pub const ListView = union(enum) {
    few: std.BoundedArray(Ast.Node.Index, 2),
    many: struct { Ast.Node.Index, Ast.Node.Index },

    /// Assumes `nodeKind(ast, node) == .list or nodeKind(ast, node) == .empty_tuple`.
    pub inline fn from(ast: Ast, node: Ast.Node.Index) ListView {
        const node_kind = nodeKind(ast, node) catch unreachable;
        switch (node_kind) {
            .empty_tuple => return .{ .few = .{} },
            .list => {},
            else => unreachable,
        }
        const nodes_tag: []const Ast.Node.Tag = ast.nodes.items(.tag);
        const nodes_data: []const Ast.Node.Data = ast.nodes.items(.data);
        return switch (nodes_tag[node]) {
            .array_init_dot_two,
            .array_init_dot_two_comma,
            => blk: {
                const data = nodes_data[node];
                assert(data.lhs != 0);
                var node_list: std.BoundedArray(Ast.Node.Index, 2) = .{};
                node_list.appendAssumeCapacity(data.lhs);
                if (data.rhs != 0) node_list.appendAssumeCapacity(data.rhs);
                break :blk .{ .few = node_list };
            },
            .array_init_dot,
            .array_init_dot_comma,
            => blk: {
                const data = nodes_data[node];
                break :blk .{ data.lhs, data.rhs };
            },
            .array_init,
            .array_init_comma,
            => unreachable,
            else => unreachable,
        };
    }

    pub inline fn elementNodeList(lv: *const ListView, ast: Ast) []const Ast.Node.Index {
        return switch (lv.*) {
            .few => |*barray| barray.constSlice(),
            .many => |range| blk: {
                const start, const end = range;
                break :blk ast.extra_data[start..end];
            },
        };
    }
};

pub const RecordView = union(enum) {
    few: std.BoundedArray(Ast.Node.Index, 2),
    many: struct { Ast.Node.Index, Ast.Node.Index },

    /// Assumes `nodeKind(ast, node) == .record or nodeKind(ast, node) == .empty_tuple`.
    pub inline fn from(ast: Ast, node: Ast.Node.Index) RecordView {
        const node_kind = nodeKind(ast, node) catch unreachable;
        switch (node_kind) {
            .empty_tuple => return .{ .few = .{} },
            .record => {},
            else => unreachable,
        }
        const nodes_tag: []const Ast.Node.Tag = ast.nodes.items(.tag);
        const nodes_data: []const Ast.Node.Data = ast.nodes.items(.data);
        return switch (nodes_tag[node]) {
            .struct_init_dot_two,
            .struct_init_dot_two_comma,
            => blk: {
                const data = nodes_data[node];
                assert(data.lhs != 0);
                var node_list: std.BoundedArray(Ast.Node.Index, 2) = .{};
                node_list.appendAssumeCapacity(data.lhs);
                if (data.rhs != 0) node_list.appendAssumeCapacity(data.rhs);
                break :blk .{ .few = node_list };
            },

            .struct_init_dot,
            .struct_init_dot_comma,
            => blk: {
                const data = nodes_data[node];
                break :blk .{ .many = .{ data.lhs, data.rhs } };
            },

            .struct_init,
            .struct_init_comma,
            => unreachable,

            else => unreachable,
        };
    }

    pub const Field = struct {
        name: []const u8,
        /// The node representing the field value.
        value: Ast.Node.Index,

        pub inline fn destructure(field: Field) struct { []const u8, Ast.Node.Index } {
            return .{ field.name, field.value };
        }
    };

    pub inline fn fieldCount(rv: *const RecordView) usize {
        return switch (rv.*) {
            .few => |barray| barray.len,
            .many => |range| blk: {
                const start, const end = range;
                break :blk end - start;
            },
        };
    }

    /// List of nodes representing each of the record's fields' values.
    pub inline fn valueNodeList(rv: *const RecordView, ast: Ast) []const Ast.Node.Index {
        return switch (rv.*) {
            .few => |*barray| barray.constSlice(),
            .many => |range| blk: {
                const start, const end = range;
                break :blk ast.extra_data[start..end];
            },
        };
    }

    /// Assumes `index < rv.fieldCount()`.
    pub fn getNameAt(rv: *const RecordView, ast: Ast, index: usize) []const u8 {
        const node = rv.getValueAt(ast, index);
        const name_tok = getFieldNameIndexFromValueNode(ast, node);
        return ast.tokenSlice(name_tok);
    }

    /// Assumes `index < rv.fieldCount()`.
    pub inline fn getValueAt(rv: *const RecordView, ast: Ast, index: usize) Ast.Node.Index {
        return rv.valueNodeList(ast)[index];
    }

    /// Assumes `index < rv.fieldCount()`.
    pub inline fn getFieldAt(rv: *const RecordView, ast: Ast, index: usize) Field {
        return .{
            .name = rv.getNameAt(ast, index),
            .value = rv.getValueAt(ast, index),
        };
    }

    pub inline fn fieldIterator(rv: *const RecordView, ast: Ast) FieldIterator {
        return .{
            .ast = ast,
            .vnl = rv.valueNodeList(ast),
            .i = 0,
        };
    }
    pub const FieldIterator = struct {
        ast: Ast,
        vnl: []const Ast.Node.Index,
        i: usize,

        pub inline fn next(iter: *FieldIterator) ?Field {
            if (iter.i == iter.vnl.len) return null;
            const value_node = iter.vnl[iter.i];
            const name_tok = getFieldNameIndexFromValueNode(iter.ast, value_node);
            return .{
                .name = iter.ast.tokenSlice(name_tok),
                .value = value_node,
            };
        }
    };

    fn getFieldNameIndexFromValueNode(ast: Ast, value_node: Ast.Node.Index) Ast.TokenIndex {
        const first_tok = ast.firstToken(value_node);
        const eql_tok = first_tok - 1;
        assert(ast.tokens.items(.tag)[eql_tok] == .equal);
        const name_tok = eql_tok - 1;
        return name_tok;
    }
};

test RecordView {
    const allocator = std.testing.allocator;

    var ast = try Ast.parse(allocator, ".{ .foo = 3, .bar = 2.0 }", .zon);
    defer ast.deinit(allocator);

    const root_expr = rootExpr(ast);
    try std.testing.expectEqual(.record, nodeKind(ast, root_expr));
    const rv = RecordView.from(ast, root_expr);

    try std.testing.expectEqual(2, rv.fieldCount());
    const name1, const value1 = rv.getFieldAt(ast, 0).destructure();
    const name2, const value2 = rv.getFieldAt(ast, 1).destructure();

    try std.testing.expectEqualStrings("foo", name1);
    try testNumberAst("3", ast, value1);

    try std.testing.expectEqualStrings("bar", name2);
    try testNumberAst("2.0", ast, value2);
}

fn testPrint(comptime fmt: []const u8, args: anytype) void {
    if (@inComptime()) {
        @compileError(std.fmt.comptimePrint(fmt, args));
    } else if (std.testing.backend_can_print) {
        std.debug.print(fmt, args);
    }
}
