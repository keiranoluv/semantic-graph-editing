"""Non-executing pickle opcode reader. Never import globals or call constructors.

Only the observed primitive/NetworkX state grammar is accepted. NEWOBJ/BUILD
produce inert dictionaries, not Python instances. REDUCE and persistent IDs are
forbidden. The CLI additionally runs in a chroot, without privileges, filesystem
or network syscalls, with CPU/address-space limits. Input/output are pipes only.
"""

import ctypes
import errno
import io
import json
import os
import pickletools
import resource
import sys
from collections import Counter
from dataclasses import dataclass, field

GLOBALS = {
    "networkx.classes.graph Graph",
    "networkx.classes.digraph DiGraph",
    "networkx.classes.reportviews NodeView",
    "builtins dict",
}

MARK = object()


@dataclass
class Symbol:
    name: str
    state: dict = field(default_factory=dict)


def parse_record(ops):
    stack, memo = [], {}

    def marked():
        index = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is MARK)
        values = stack[index + 1 :]
        del stack[index:]
        return values

    def put(table, values):
        if type(table) is not dict or len(values) % 2:
            raise ValueError("invalid_dictionary")
        for k, v in zip(values[::2], values[1::2], strict=True):
            if type(k) not in (str, int):
                raise ValueError("invalid_dictionary_key")
            if k in table:
                raise ValueError("repeated_dictionary_key")
            table[k] = v

    for op, arg, _ in ops:
        name = op.name
        if name in ("PROTO", "FRAME"):
            continue
        if name in ("GLOBAL",):
            if arg not in GLOBALS:
                raise ValueError("forbidden_global:" + arg)
            stack.append(Symbol(arg))
        elif name == "STACK_GLOBAL":
            symbol, module = stack.pop(), stack.pop()
            key = module + " " + symbol
            if key not in GLOBALS:
                raise ValueError("forbidden_global:" + key)
            stack.append(Symbol(key))
        elif name == "NEWOBJ":
            args, symbol = stack.pop(), stack.pop()
            if args != () or not isinstance(symbol, Symbol) or symbol.name == "builtins dict":
                raise ValueError("invalid_newobj")
            stack.append(Symbol(symbol.name))
        elif name == "BUILD":
            state = stack.pop()
            if type(state) is not dict or not isinstance(stack[-1], Symbol) or stack[-1].state:
                raise ValueError("invalid_build")
            stack[-1].state = state
        elif name in ("BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"):
            key = len(memo) if name == "MEMOIZE" else int(arg)
            if key in memo:
                raise ValueError("repeated_memo")
            memo[key] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo[int(arg)])
        elif name == "MARK":
            stack.append(MARK)
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BININT", "BININT1", "BININT2", "LONG1"):
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name in ("NEWTRUE", "NEWFALSE"):
            stack.append(name == "NEWTRUE")
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            put(stack[-1], [key, value])
        elif name == "SETITEMS":
            values = marked()
            put(stack[-1], values)
        elif name == "TUPLE":
            stack.append(tuple(marked()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            size = int(name[-1])
            values = tuple(stack[-size:])
            del stack[-size:]
            stack.append(values)
        elif name == "APPEND":
            value = stack.pop()
            stack[-1].append(value)
        elif name == "APPENDS":
            values = marked()
            stack[-1].extend(values)
        elif name == "STOP":
            if len(stack) != 1 or not isinstance(stack[0], Symbol):
                raise ValueError("invalid_root")
            return graph_state(stack[0])
        else:
            raise ValueError("forbidden_opcode:" + name)
    raise ValueError("missing_stop")


def graph_state(symbol):
    if symbol.name not in ("networkx.classes.graph Graph", "networkx.classes.digraph DiGraph"):
        raise ValueError("not_networkx_graph")
    state = symbol.state
    allowed = {"graph", "_node", "_adj", "_succ", "_pred", "nodes", "__networkx_cache__"}
    for key, value in state.items():
        if key.endswith("_factory"):
            if not isinstance(value, Symbol) or value.name != "builtins dict":
                raise ValueError("unknown_factory")
        elif key not in allowed:
            raise ValueError("unknown_graph_state:" + key)
    if state.get("graph") != {}:
        raise ValueError("nonempty_graph_metadata")
    if state.get("__networkx_cache__", {}) != {}:
        raise ValueError("nonempty_networkx_cache")
    nodes, adjacency = state["_node"], state.get("_succ", state["_adj"])
    if type(nodes) is not dict or type(adjacency) is not dict or set(adjacency) != set(nodes):
        raise ValueError("invalid_adjacency")
    directed = symbol.name.endswith(" DiGraph")
    if "nodes" in state:
        view = state["nodes"]
        if (
            not isinstance(view, Symbol)
            or view.name != "networkx.classes.reportviews NodeView"
            or set(view.state) != {"_nodes"}
            or view.state["_nodes"] is not nodes
        ):
            raise ValueError("inconsistent_cached_node_view")
    rows = []
    for identifier, data in nodes.items():
        if type(identifier) is not int or type(data) is not dict or set(data) != {"feature"}:
            raise ValueError("unknown_node_schema")
        if type(data["feature"]) is not str or not data["feature"].strip():
            raise ValueError("invalid_node_feature")
        rows.append({"id": identifier, "data": data})
    edges, seen = [], set()
    for u, neighbors in adjacency.items():
        for v, data in neighbors.items():
            if v not in nodes or type(data) is not dict or set(data) != {"feature"}:
                raise ValueError("unknown_edge_schema")
            if type(data["feature"]) is not str or not data["feature"].strip():
                raise ValueError("invalid_edge_feature")
            if not directed and adjacency[v].get(u) != data:
                raise ValueError("asymmetric_undirected_adjacency")
            key = (u, v) if directed else tuple(sorted((u, v)))
            if key not in seen:
                edges.append({"u": key[0], "v": key[1], "data": data})
                seen.add(key)
    if directed:
        expected = {n: {} for n in nodes}
        for e in edges:
            expected[e["v"]][e["u"]] = e["data"]
        if state.get("_pred") != expected:
            raise ValueError("inconsistent_predecessors")
    return {"directed": directed, "nodes": rows, "edges": edges, "graph_metadata": {}}


def scan(content, indices):
    stream, selected, counts, index = io.BytesIO(content), {}, Counter(), 0
    wanted = set(indices)
    while stream.tell() < len(content):
        ops = []
        start = stream.tell()
        for op, arg, position in pickletools.genops(stream):
            counts[op.name] += 1
            if index in wanted:
                ops.append((op, arg, position))
            if stream.tell() - start > 1_000_000:
                raise ValueError("record_byte_budget")
        if index in wanted:
            try:
                selected[str(index)] = {"graph": parse_record(ops), "error": None}
            except (ValueError, KeyError, IndexError, TypeError, StopIteration) as error:
                selected[str(index)] = {"graph": None, "error": f"schema:{error}"}
        index += 1
    if wanted - set(range(index)):
        raise ValueError("missing_selected_records")
    return {"record_count": index, "selected": selected, "opcode_counts": dict(counts)}


def confine(jail):
    """Linux, fail closed. No host paths or credentials remain accessible."""
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.clear()
    os.chroot(jail)
    os.chdir("/")
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    ctx = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not ctx:
        raise RuntimeError("seccomp_init_failed")
    for name in (
        "read",
        "write",
        "close",
        "fstat",
        "newfstatat",
        "lseek",
        "brk",
        "mmap",
        "mremap",
        "munmap",
        "mprotect",
        "futex",
        "clock_gettime",
        "getrusage",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "sigaltstack",
        "exit",
        "exit_group",
    ):
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number < 0 or lib.seccomp_rule_add(ctx, 0x7FFF0000, number, 0) != 0:
            raise RuntimeError("seccomp_rule_failed")
    if lib.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp_load_failed")


if __name__ == "__main__":
    indices = json.loads(sys.argv[2])
    confine(sys.argv[1])
    data = sys.stdin.buffer.read(32 * 1024**2 + 1)
    if len(data) > 32 * 1024**2:
        raise ValueError("input_byte_budget")
    result = scan(data, indices)
    result["peak_worker_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["isolation"] = "empty_chroot_uid65534_seccomp_allowlist_no_open_socket_exec_512MiB_60s"
    print(json.dumps(result, allow_nan=False))
