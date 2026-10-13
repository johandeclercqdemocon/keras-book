"""Audit the prose of each chapter against the book's own outputs.

    python3 scripts/audit_numbers.py            # numbers in prose that no output, code or exercise contains
    python3 scripts/audit_numbers.py --tables   # parameter tables against the printed signatures

Numbers: every number in prose (outside code fences and inline code) must appear in an output block, a code
block, or the chapter's exercises and solutions. A number that appears nowhere is printed with its
sentence, and a person decides: derived (check the arithmetic), copied wrong (fix it), or invented (delete).

Tables: a row such as | `fit(batch_size)` | `None` | ... | is checked against the signature that the chapter
printed. The parameter must exist, and its default must be the printed default.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)
NUMBER = re.compile(r"(?<![\w.`/-])(\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)(?![\w`])")
SKIP_SMALL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "16"}  # layer sizes, chapter numbers, list items


def chapter_files():
    return sorted((ROOT / "chapters").glob("[0-9][0-9]-*.md"))


def related(chapter):
    number = chapter.name[:2]
    return [chapter, ROOT / "exercises" / f"{number}.md", ROOT / "exercises" / "solutions" / f"{number}.md"]


def prose_of(text):
    text = FENCE.sub("", text)
    text = re.sub(r"`[^`\n]+`", "", text)              # inline code
    text = re.sub(r"\]\([^)]*\)", "]", text)           # link targets
    return text


def normalise(token):
    return token.replace(",", "")


def audit_numbers():
    problems = 0
    for chapter in chapter_files():
        corpus = ""
        for path in related(chapter):
            if path.exists():
                corpus += "\n".join(body for _, body in FENCE.findall(path.read_text(encoding="utf-8")))
        corpus_numbers = {normalise(m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", corpus)}
        header_stop = prose_of(chapter.read_text(encoding="utf-8"))
        for sentence in re.split(r"(?<=[.!?])\s+|\n\n", header_stop):
            for token in NUMBER.findall(sentence):
                value = normalise(token)
                if value in SKIP_SMALL or value in corpus_numbers:
                    continue
                if "." in value and value.rstrip("0").rstrip(".") in corpus_numbers:
                    continue
                # numbers like 0.9711 are also matched when the output prints 0.97110
                problems += 1
                print(f"{chapter.name}: {token:>10s}  |  {' '.join(sentence.split())[:150]}")
    return problems


def printed_signatures(text):
    """Signature lines printed by `show(...)` blocks: the text block that follows a python block with show(."""
    joined = []
    for match in re.finditer(r"```python\n(.*?)```\s*```text\n(.*?)```", text, re.S):
        if "show(" in match.group(1):
            joined.append(re.sub(r"\n\s+", " ", match.group(2)))
    return "\n".join(joined)


def audit_tables():
    problems = 0
    row = re.compile(r"^\|\s*`([A-Za-z_.]+)\(([^`]*)\)`[^|]*\|\s*([^|]*?)\s*\|", re.M)
    for chapter in chapter_files():
        text = chapter.read_text(encoding="utf-8")
        signatures = printed_signatures(text)
        if not signatures:
            continue
        for name, params, default in row.findall(text):
            function = name.split(".")[-1]
            found = re.search(rf"(?:^|[\s.]){re.escape(function)}\((.*)$", signatures, re.M)
            if not found:
                continue          # a method that the reader writes (call, update_state): no printed signature
            body = found.group(1)
            for param in [p.strip() for p in params.split(",") if p.strip()]:
                param = param.split("=")[0].strip()
                if not re.fullmatch(r"[A-Za-z_]\w*", param):
                    continue
                match = re.search(rf"(?:^|[(,\s]){re.escape(param)}(?::\s*\w+)?(?:\s*=\s*([^,)]+|\([^)]*\)))?(?=[,)]|$)", body)
                if not match:
                    problems += 1
                    print(f"{chapter.name}: {name}({param}) is not in the printed signature of {function}")
                    continue
                printed = match.group(1)
                wanted = default.strip("` ").strip("'")
                if wanted in ("—", "") or wanted.startswith("all `None"):
                    continue
                if wanted == "required":
                    if printed is not None:
                        problems += 1
                        print(f"{chapter.name}: {name}({param}) is marked required, the signature has {printed}")
                elif printed is None or printed.strip("'") != wanted:
                    problems += 1
                    print(f"{chapter.name}: {name}({param}) says {wanted!r}, the signature prints {printed!r}")
    return problems


if __name__ == "__main__":
    count = audit_tables() if "--tables" in sys.argv else audit_numbers()
    print(f"\n{count} item(s) for review.")
