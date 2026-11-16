"""D14: she reads her own code (read-only) to point at the line that failed.
D22: she never changes it. The probes below try, and must fail; the writer inventory fails the day a
tool of hers learns to write files without being registered here."""
import ast
import builtins
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from aerys_v2.tools import my_code
from aerys_v2.tools.my_code import build_read_my_code_tool, package_root

OWNER = "00000000-0000-0000-0000-0000000000c1"
ROOT = package_root()
REPO = ROOT.parent.parent


def runtime(user_id=OWNER, **extra):
    return SimpleNamespace(config={"configurable": {"identity": {"user_id": user_id, **extra}, "thread_id": "t"}})


def call(tool, rt=None, **kw):
    return tool.func(runtime=rt or runtime(), **kw)


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" not in f.parts:
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def test_search_names_file_and_line_then_read_shows_the_lines():
    tool = build_read_my_code_tool(OWNER)
    hits = call(tool, search_for="def build_read_my_code_tool")
    m = re.search(r"tools/my_code\.py:(\d+): def build_read_my_code_tool", hits)
    assert m, hits
    page = call(tool, path="tools/my_code.py", line=int(m.group(1)))
    assert page.startswith("tools/my_code.py, lines ")
    assert f"{int(m.group(1)):>5}| def build_read_my_code_tool" in page


@pytest.mark.parametrize("path", ["tools/my_code.py", "aerys_v2/tools/my_code.py", "src/aerys_v2/tools/my_code.py"])
def test_the_path_forms_she_will_use(path):
    assert call(build_read_my_code_tool(OWNER), path=path).startswith("tools/my_code.py, lines 1-")


@pytest.mark.parametrize("path", [
    "../../tests/test_my_code.py",          # a real .py file, outside her package
    "tools/../../../tests/test_my_code.py",
    "../../pyproject.toml",
    str(REPO / "tests" / "test_my_code.py"),  # absolute
    "/etc/passwd",
    "C:/Windows/win.ini",
    ".env",
    "../../.env",
    "",
])
def test_nothing_outside_her_own_code(path):
    out = call(build_read_my_code_tool(OWNER), path=path) if path else my_code.read(ROOT, path)
    assert "isn't one of my code files" in out


def test_a_symlink_out_of_her_code_is_refused(tmp_path):
    root = tmp_path / "aerys_v2"
    root.mkdir()
    (root / "real.py").write_text("x = 'mine'\n")
    outside = tmp_path / "secret.py"
    outside.write_text("TOKEN = 'not hers'\n")
    (root / "link.py").symlink_to(outside)
    tool = build_read_my_code_tool(OWNER, root=root.resolve())
    assert "isn't one of my code files" in call(tool, path="link.py")
    assert "not hers" not in call(tool, search_for="TOKEN")
    assert "real.py:1" in call(tool, search_for="mine")


def test_it_only_ever_opens_files_to_read(monkeypatch):
    modes, real_open = [], builtins.open

    def spy(file, mode="r", *a, **kw):
        modes.append(mode)
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    tool = build_read_my_code_tool(OWNER)
    call(tool, search_for="read_my_code")
    call(tool, path="factory.py", line=10)
    assert modes and set(modes) <= {"r", "rt"}


@pytest.mark.parametrize("rt, ok", [
    (runtime(), True),
    (runtime(platform="wake", unattended=True), True),                    # her own wake-up runs as him
    (runtime(user_id="00000000-0000-0000-0000-0000000000b2"), False),     # anyone else
    (SimpleNamespace(config={}), False),                                  # no identity at all
])
def test_only_his_turns_and_her_own_wake_ups(rt, ok):
    out = call(build_read_my_code_tool(OWNER), rt, path="tools/timer.py")
    assert out.startswith("tools/timer.py, lines ") is ok
    assert out.startswith("Not available: only Chris's turns") is not ok


def test_owner_side_only_never_guests():
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for

    s = Settings(_env_file=None, anthropic_api_key="sk-test", owner_person_id=OWNER)
    assert "read_my_code" in {t.name for t in action_tools_for(s)}
    assert "read_my_code" not in {t.name for t in action_tools_for(s, guest=True)}


# --- D22: she never changes her own code ----------------------------------------------------------

#: Every place one of her tools writes a file, with the reason it may. Empty on 11/15: none does.
#: A new writer must be added here on purpose, and must never reach her own package.
REGISTERED_WRITERS: dict[str, str] = {}
_WRITE_METHODS = {"write_text", "write_bytes", "write", "writelines", "unlink", "mkdir", "rmdir", "rename",
                  "touch", "symlink_to", "hardlink_to", "chmod", "truncate"}
_WRITE_MODULES = {"shutil", "subprocess", "tempfile"}
_OS_WRITES = {"remove", "unlink", "rename", "renames", "replace", "makedirs", "mkdir", "rmdir", "removedirs",
              "truncate", "chmod", "system", "open", "write", "symlink", "link"}
_STREAM_MODULES = {"io", "gzip", "bz2", "lzma", "codecs", "os"}


def _writes_mode(node, index, *, strict=True):
    """True when an open() call's mode can write: w/a/x/+ in a literal mode, or a mode that isn't a literal.
    strict=False (x.open(...) on some object): a positional non-literal isn't taken for a mode, since
    tickets.open(thread_id, ...) opens a ticket, not a file; a mode= keyword still counts."""
    mode = next((k.value for k in node.keywords if k.arg == "mode"), None)
    keyword = mode is not None
    if mode is None and len(node.args) > index:
        mode = node.args[index]
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(c in mode.value for c in "wax+")
    return strict or keyword


def write_calls(source: str) -> list[int]:
    """Line numbers of every call in `source` that can write or remove a file. By syntax, not by regex, so
    a multi-line open(..., "w") and Path.replace(dest) are caught (Codex review 11/15)."""
    rows = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == "open":
            hit = _writes_mode(node, 1)
        elif isinstance(f, ast.Attribute):
            owner = f.value.id if isinstance(f.value, ast.Name) else None
            if f.attr == "open":
                stream = owner in _STREAM_MODULES
                hit = _writes_mode(node, 1 if stream else 0, strict=stream)
            elif owner == "os":
                hit = f.attr in _OS_WRITES
            elif owner in _WRITE_MODULES:
                hit = True
            elif f.attr == "replace":
                hit = len(node.args) == 1 and not node.keywords      # Path.replace(dest); str.replace takes two
            else:
                hit = f.attr in _WRITE_METHODS
        else:
            hit = False
        if hit:
            rows.append(node.lineno)
    return sorted(rows)


def writers_in(folder: Path) -> dict[str, list[int]]:
    found = {}
    for f in sorted(folder.rglob("*.py")):
        if "__pycache__" not in f.parts and (rows := write_calls(f.read_text())):
            found[str(f.relative_to(folder))] = rows
    return found


def test_no_tool_of_hers_writes_files_unless_registered():
    found = writers_in(ROOT / "tools")
    assert set(found) == set(REGISTERED_WRITERS), f"a tool now writes files; register it with a reason: {found}"


@pytest.mark.parametrize("source", [
    "def f(p):\n    p.write_text('changed')\n",
    "def f(p):\n    with open(\n        p,\n        'w',\n    ) as fh:\n        fh.write('x')\n",
    "def f(p):\n    open(str(p), 'w')\n",
    "def f(p):\n    p.open('w+b')\n",
    "def f(p, dest):\n    p.replace(dest)\n",
    "def f(p, m):\n    open(p, mode=m)\n",
    "def f(p, m):\n    p.open(mode=m)\n",
    "import io\ndef f(p):\n    io.open(p, 'a')\n",
    "import os\ndef f(p):\n    os.remove(p)\n",
    "import shutil\ndef f(a, b):\n    shutil.copy(a, b)\n",
    "import subprocess\ndef f():\n    subprocess.run(['sed', '-i', 's/a/b/', 'x.py'])\n",
])
def test_the_writer_probe_catches_every_write_form(source):
    assert write_calls(source), source


def test_the_writer_probe_leaves_reads_alone(tmp_path):
    benign = ("import json\ndef g(p, s):\n    a = open(p).read()\n    b = p.read_text()\n"
              "    with p.open() as fh:\n        c = fh.read()\n    t = tickets.open(thread, text, True)\n"
              "    return s.replace('a', 'b') + json.dumps(a)\n")
    assert write_calls(benign) == []
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "sneaky.py").write_text("def f(p):\n    p.unlink()\n")
    assert writers_in(tmp_path) == {"nested/sneaky.py": [2]}


def test_a_pipe_or_a_giant_file_is_never_opened(tmp_path):
    import os

    root = tmp_path / "aerys_v2"
    root.mkdir()
    (root / "ok.py").write_text("NAME = 'findme'\n")
    os.mkfifo(root / "stuck.py")                                   # opening this would block forever
    (root / "huge.py").write_text("findme = 1\n" * (my_code.MAX_FILE_BYTES // 11 + 10))
    tool = build_read_my_code_tool(OWNER, root=root.resolve())
    assert call(tool, search_for="findme") == "ok.py:1: NAME = 'findme'"
    assert "isn't one of my code files" in call(tool, path="stuck.py")
    assert "isn't one of my code files" in call(tool, path="huge.py")


def test_she_tries_to_change_her_code_through_her_own_engine_and_cannot():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from aerys_v2.factory import build_action_graph

    before = tree_digest(ROOT)
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "write_file", "args": {"path": "tools/my_code.py", "text": "pass"}, "id": "w"},
            {"name": "read_my_code", "args": {"path": "../../tests/test_my_code.py"}, "id": "r"}]),
        AIMessage(content="I can't change my code.")]))
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for

    # Her real owner-side tool list (with nothing else armed), not a hand-picked one (Codex review 11/15).
    tools = action_tools_for(Settings(_env_file=None, anthropic_api_key="sk-test", owner_person_id=OWNER))
    assert "read_my_code" in {t.name for t in tools}
    graph = build_action_graph(model, "test", tools)
    result = graph.invoke({"messages": [HumanMessage(content="Rewrite your own web reader")]},
                          {"configurable": {"identity": {"user_id": OWNER}, "thread_id": "probe:d22"}})
    replies = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert "write_file" in replies["w"] and "not a valid tool" in replies["w"]
    assert "isn't one of my code files" in replies["r"]
    assert tree_digest(ROOT) == before
