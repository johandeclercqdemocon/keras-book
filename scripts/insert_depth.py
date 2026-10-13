"""Insert (or replace) the `## How it works` and `## Reference` sections of one chapter.

    python3 scripts/insert_depth.py chapters/01-the-first-model.md section.md

The new text goes immediately before `## What you have now`. If the chapter already has a
`## How it works` section, everything from there up to `## What you have now` is replaced.
"""
import pathlib
import sys

chapter, section = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text, new = chapter.read_text(encoding="utf-8"), section.read_text(encoding="utf-8").rstrip() + "\n\n"
end = text.index("## What you have now")
start = text.find("## How it works")
if start != -1 and start < end:
    text = text[:start] + new + text[end:]
else:
    text = text[:end] + new + text[end:]
chapter.write_text(text, encoding="utf-8")
print("updated", chapter)
