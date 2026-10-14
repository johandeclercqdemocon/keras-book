"""Audit the prose of each chapter against the book's own outputs.

    python3 scripts/audit_numbers.py            # numbers in prose that no output, code or exercise contains

Numbers: every number in prose (outside code fences and inline code) must appear in an output block, a code
block, or the chapter's exercises and solutions. A number that appears nowhere is printed with its
sentence, and a person decides: derived (check the arithmetic), copied wrong (fix it), or invented (delete).

The parameter tables are not audited here. `scripts/check_snippets.py` checks them: each `####` block must have the
real signature, and one row with the real default for each parameter.
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


if __name__ == "__main__":
    count = audit_numbers()
    print(f"\n{count} item(s) for review.")
