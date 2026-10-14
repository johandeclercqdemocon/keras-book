"""Check that this book's links, code and claimed outputs still agree.

Run it with the interpreter that has Keras installed, for example `uv run python scripts/check_snippets.py`.
Set KERAS_BOOK_JOBS to change the number of parallel snippets (default 6: each one starts a TensorFlow process).

    python3 scripts/check_snippets.py            # check everything
    python3 scripts/check_snippets.py --fast     # skip the blocks: links, contracts, sections, signatures, names
    python3 scripts/check_snippets.py --fill     # write real output into `text` blocks marked "# fill"
    python3 scripts/check_snippets.py --refill chapters/16-hashing.md   # regenerate every checked output in a file
    python3 scripts/check_snippets.py --signatures   # write the real signature into blocks that hold "# fill"

Checks, all mechanical and all zero-false-positive by design:

  links     every relative markdown link resolves to a file that exists
  contract  every chapter carries Build / Theory kept / Source / Run this first
  python    every ```python block -- in chapters, exercises AND solutions -- is executed, exit 0
  output    every ```text block directly below a ```python block must equal the real output
  examples  every examples/*.py module named in a chapter runs and exits 0
  sections  every chapter has `## How it works` and `## Reference`
  params    every `#### keras.name` heading: the ```text block under it is the real signature of the
            installed Keras, every table row is a real parameter with its real default, every parameter
            has a row (`*args` and `**kwargs` included, with the default `—`), and no callable has two
            blocks. Every name in a Reference index has a `####` block somewhere in the book
  names     every `keras.name` that the text mentions exists in the installed Keras

A reader copies a block and runs it, so the check is execution, not syntax. Blocks that are not
programs belong in a ```text, ```bash or ```json fence and are skipped. A ```text block that
starts with `# output varies` is skipped (timings). Never weaken this file to fit a snippet --
rewrite the snippet.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)#]+?)(?:#[^)]*)?\)")
PY_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
PAIRED = re.compile(r"(```python\n((?:(?!```)[\s\S])*)```\s*)```text\n((?:(?!```)[\s\S])*)```")
CONTRACT = ("> **Build**", "> **Theory kept**", "> **Source**", "> **Run this first**")
RUN_TIMEOUT = 900
JOBS = int(os.environ.get("KERAS_BOOK_JOBS", "6"))
BOOK_ENV = {"KERAS_BACKEND": "tensorflow", "TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0",
            "PYTHONWARNINGS": "ignore", "CUDA_VISIBLE_DEVICES": "",
            # Without this, training a CNN on the CPU is not repeatable: see Chapter 15.
            "TF_DETERMINISTIC_OPS": "1"}
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"[FAIL] {msg}")


def md_files() -> list[pathlib.Path]:
    return sorted(p for p in [ROOT / "README.md", *(ROOT / "chapters").glob("*.md"), *(ROOT / "appendices").glob("*.md"),
                   *(ROOT / "exercises").glob("*.md"), *(ROOT / "exercises" / "solutions").glob("*.md"),
                   ] if p.exists())


def run(src: str):
    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "snippet.py"
        f.write_text(src, encoding="utf-8")
        env = {**os.environ, **BOOK_ENV, "PYTHONPATH": str(ROOT), "PYTHONHASHSEED": "0"}
        return subprocess.run([sys.executable, str(f)], capture_output=True, text=True,
                              timeout=RUN_TIMEOUT, cwd=ROOT, env=env)


def check_links(files):
    n = 0
    for p in files:
        for t in LINK.findall(p.read_text(encoding="utf-8")):
            if t.startswith(("http://", "https://", "mailto:")):
                continue
            n += 1
            if not (p.parent / t).resolve().exists():
                fail(f"{p.relative_to(ROOT)} -> {t} does not exist")
    return n


def check_contracts():
    chs = sorted((ROOT / "chapters").glob("*.md"))
    for p in chs:
        text = p.read_text(encoding="utf-8")
        miss = [m for m in CONTRACT if m not in text]
        if miss:
            fail(f"{p.relative_to(ROOT)} missing {', '.join(miss)}")
    return len(chs)


def norm(t):
    return [l.rstrip() for l in t.rstrip().splitlines()]


_cache: dict[str, object] = {}


def run_cached(src: str):
    """Run each distinct snippet once, however many checks need its result."""
    if src not in _cache:
        try:
            _cache[src] = run(src)
        except subprocess.TimeoutExpired:
            _cache[src] = None
    return _cache[src]


def check_python_and_outputs(files):
    """Execute every ```python block once (in parallel); compare each paired ```text block with its stdout."""
    from concurrent.futures import ThreadPoolExecutor

    blocks, paired = [], {}
    for p in files:
        text = p.read_text(encoding="utf-8")
        for i, src in enumerate(PY_BLOCK.findall(text), 1):
            blocks.append((p, i, src))
        for j, m in enumerate(PAIRED.finditer(text), 1):
            paired[(p, m.group(2))] = (j, m.group(3))

    for p, i, src in blocks:
        try:
            ast.parse(src)
        except SyntaxError as e:
            fail(f"{p.relative_to(ROOT)} block {i} does not parse: {e}")

    unique = sorted({src for _, _, src in blocks})
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        list(pool.map(run_cached, unique))

    checked = 0
    for p, i, src in blocks:
        label = f"{p.relative_to(ROOT)} block {i}"
        d = _cache[src]
        if d is None:
            fail(f"{label} timed out after {RUN_TIMEOUT}s")
        elif d.returncode:
            fail(f"{label} failed: {(d.stderr.strip().splitlines() or ['?'])[-1]}")
        elif (p, src) in paired:
            j, want = paired[(p, src)]
            if want.lstrip().startswith("# output varies"):
                continue
            checked += 1
            a, c = norm(d.stdout), norm(want)
            if a != c:
                for k, (g, w) in enumerate(zip(a, c), 1):
                    if g != w:
                        fail(f"{label} output line {k}:\n      claimed: {w}\n      actual:  {g}")
                        break
                else:
                    fail(f"{label} output: claimed {len(c)} lines, produced {len(a)}")
    return len(blocks), checked


def check_examples(files=None):
    """Run the example modules that the chapters name. With files, only those that the given chapters name."""
    chapters = [p for p in (files or (ROOT / "chapters").glob("*.md")) if p.parent.name == "chapters"]
    mods = sorted(set(re.findall(r"python3? -m (examples\.[A-Za-z0-9_]+)",
                                 "".join(p.read_text(encoding="utf-8") for p in chapters))))
    for m in mods:
        if not (ROOT / (m.replace(".", "/") + ".py")).exists():
            fail(f"{m} named in a chapter but file is missing")
            continue
        try:
            d = subprocess.run([sys.executable, "-m", m], capture_output=True, text=True, timeout=RUN_TIMEOUT,
                               cwd=ROOT, env={**os.environ, **BOOK_ENV, "PYTHONPATH": str(ROOT)})
        except subprocess.TimeoutExpired:
            fail(f"{m} timed out")
            continue
        if d.returncode:
            fail(f"{m} exited {d.returncode}: {(d.stderr.strip().splitlines() or ['?'])[-1]}")
    return len(mods)


def check_sections():
    """Every chapter has a How it works section and a Reference section."""
    chs = sorted((ROOT / "chapters").glob("*.md"))
    for p in chs:
        text = p.read_text(encoding="utf-8")
        for heading in ("## How it works", "## Reference"):
            if heading not in text:
                fail(f"{p.relative_to(ROOT)} has no '{heading}' section")
    return len(chs)


HEADING = re.compile(r"^#### `([A-Za-z_][\w.]*)`\s*$", re.M)
ROW = re.compile(r"^\| ((?:`[^`]+`(?:, )?)+) \| (required|`[^`]*`|—) \|", re.M)
SIGBLOCK = re.compile(r"\A\s*```text\n(.*?)```", re.S)
INDEX_ROW = re.compile(r"^\| `([A-Za-z_][\w.]*)` \|", re.M)


def _sections(text: str):
    """Yield (name, body) for each `#### name` heading, up to the next heading of any level.

    Lines inside a fenced code block are never headings, so a `# comment` in code does not end a section.
    """
    lines = text.splitlines(keepends=True)
    in_fence, pos, heads = False, 0, []
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(r"#{1,6} ", line):
            heads.append((pos, line))
        pos += len(line)
    for k, (off, line) in enumerate(heads):
        m = HEADING.match(line.rstrip("\n"))
        if m:
            end = heads[k + 1][0] if k + 1 < len(heads) else len(text)
            yield m.group(1), text[off + len(line):end]


def _reference(text: str) -> str:
    m = re.search(r"^## Reference\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def _keras():
    for k, v in BOOK_ENV.items():
        os.environ.setdefault(k, v)
    sys.path.insert(0, str(ROOT))
    from examples.signatures import parameters, resolve, signature_text
    return parameters, resolve, signature_text


def check_parameters():
    """See `params` in the docstring of this file."""
    parameters, resolve, signature_text = _keras()
    checked, defined = 0, {}
    for p in sorted((ROOT / "chapters").glob("*.md")):
        text = p.read_text(encoding="utf-8")
        for name, body in _sections(text):
            where = f"{p.relative_to(ROOT)} `{name}`"
            if name in defined:
                fail(f"{where}: also has a block in {defined[name]}. Link to that block instead")
            defined.setdefault(name, p.name)
            try:
                obj = resolve(name)
                real, sig = dict(parameters(obj)), signature_text(obj, name)
            except Exception as e:  # noqa: BLE001
                fail(f"{where}: cannot find this callable ({type(e).__name__}: {e})")
                continue
            m = SIGBLOCK.search(body)
            if not m:
                fail(f"{where}: no ```text signature block directly under the heading")
            elif m.group(1).strip() != sig:
                fail(f"{where}: signature block differs from the installed signature (run --signatures)")
            seen = []
            for r in ROW.finditer(body):
                default = r.group(2)
                for pname in [n.strip("` ") for n in r.group(1).split(", ")]:
                    seen.append(pname)
                    if pname not in real:
                        fail(f"{where}: `{pname}` is not a parameter")
                        continue
                    if real[pname] == "—":
                        if default != "—":
                            fail(f"{where}: `{pname}` has no default, the table says {default}")
                        continue
                    if default == "—":
                        fail(f"{where}: `{pname}` has the default {real[pname]}, the table says —")
                        continue
                    want = "required" if real[pname] is None else f"`{real[pname]}`"
                    if default != want:
                        fail(f"{where}: default of `{pname}` is {want}, the table says {default}")
            if not seen and real:
                fail(f"{where}: no parameter table")
            for dup in sorted({n for n in seen if seen.count(n) > 1}):
                fail(f"{where}: `{dup}` is listed twice")
            for pname in [n for n in real if n not in seen]:
                fail(f"{where}: `{pname}` is in the signature and has no row in the table")
            checked += 1
    for p in sorted((ROOT / "chapters").glob("*.md")):
        for name in INDEX_ROW.findall(_reference(p.read_text(encoding="utf-8"))):
            if name not in defined:
                fail(f"{p.relative_to(ROOT)}: Reference names `{name}`, which has no `####` block in the book")
    return checked


def fill_signatures(files):
    """Write the real signature into every signature block that holds only `# fill`."""
    _, resolve, signature_text = _keras()
    for p in files:
        if p.parent.name != "chapters":
            continue
        text = p.read_text(encoding="utf-8")
        out, pos = [], 0
        for h in HEADING.finditer(text):
            m = re.compile(r"(\s*```text\n)# fill\n```").match(text, h.end())
            if m:
                out.append(text[pos:h.end()])
                out.append(m.group(1) + signature_text(resolve(h.group(1)), h.group(1)) + "\n```")
                pos = m.end()
        out.append(text[pos:])
        new = "".join(out)
        if new != text:
            p.write_text(new, encoding="utf-8")
            print(f"signatures filled in {p.relative_to(ROOT)}")


KERAS_NAME = re.compile(r"`(keras\.[A-Za-z_][\w.]*)")


def check_identifiers():
    """Every `keras.name` that the text mentions must exist in the installed Keras."""
    _, resolve, _ = _keras()
    seen = set()
    for p in md_files():
        for name in KERAS_NAME.findall(p.read_text(encoding="utf-8")):
            name = name.rstrip(".")
            if name in seen:
                continue
            seen.add(name)
            try:
                resolve(name)
            except (AttributeError, ImportError):
                fail(f"{p.relative_to(ROOT)}: `{name}` does not exist in this Keras")
    return len(seen)


def fill(files, refill=False):
    """Write real stdout into ```text blocks: those marked "# fill", or (refill) every checked block."""
    from concurrent.futures import ThreadPoolExecutor

    body = r"[\s\S]*?" if refill else r"# fill\n"
    marker = re.compile(r"(```python\n((?:(?!```)[\s\S])*)```\s*```text\n)(?!# output varies)" + body + "```")
    todo = {m.group(2) for p in files for m in marker.finditer(p.read_text(encoding="utf-8"))}
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        list(pool.map(run_cached, sorted(todo)))
    for p in files:
        text = p.read_text(encoding="utf-8")

        def sub(m):
            d = run_cached(m.group(2))
            if d is None or d.returncode:
                raise SystemExit(f"{p}: snippet failed:\n{d.stderr if d else 'timeout'}")
            return m.group(1) + d.stdout.rstrip() + "\n```"

        new = marker.sub(sub, text)
        if new != text:
            p.write_text(new, encoding="utf-8")
            print(f"filled {p.relative_to(ROOT)}")


def main():
    files = md_files()
    only = [pathlib.Path(a).resolve() for a in sys.argv[1:] if not a.startswith("--")]
    if only:
        files = [f for f in files if f.resolve() in only]
    if "--signatures" in sys.argv:
        fill_signatures(files)
        return 0
    if "--fill" in sys.argv or "--refill" in sys.argv:
        fill(files, refill="--refill" in sys.argv)
        return 0
    links, chs = check_links(files), check_contracts()
    secs, sigs, names = check_sections(), check_parameters(), check_identifiers()
    if "--fast" in sys.argv:
        blocks = outs = mods = 0
    else:
        (blocks, outs), mods = check_python_and_outputs(files), check_examples(files if only else None)
    print(f"[ OK ] {links} relative links resolve")
    print(f"[ OK ] {chs} chapters carry their contract")
    print(f"[ OK ] {blocks} python blocks run")
    print(f"[ OK ] {outs} claimed outputs match what the code prints")
    print(f"[ OK ] {mods} example modules run")
    print(f"[ OK ] {secs} chapters have How it works and Reference")
    print(f"[ OK ] {sigs} callables have their real signature, real defaults and a row for every parameter")
    print(f"[ OK ] {names} distinct keras.* names in the text exist")
    if failures:
        print(f"\n{len(failures)} problem(s).")
        return 1
    print(f"\nAll checks passed across {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
