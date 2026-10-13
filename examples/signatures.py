"""Print the signature of Keras callables, as the installed version defines them.

The chapters call `show(...)` in a block, and the checker compares the printed text with the text
block under it. A signature in the book is therefore produced by the library, not typed by a person.

Each argument is a callable, or a pair `(label, callable)` when the qualified name of the callable is
an implementation class (Keras puts `Model.fit` on a back-end `Trainer` class).
"""
import inspect
import textwrap


def signature_of(fn, label=None) -> str:
    """`label(parameters)` with a leading `self` removed and the line wrapped at 78 characters."""
    parameters = list(inspect.signature(fn).parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    text = ", ".join(str(p) for p in parameters)
    name = label or getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    return textwrap.fill(f"{name}({text})", width=78, subsequent_indent="    ",
                         break_long_words=False, break_on_hyphens=False)


def show(*items) -> None:
    for item in items:
        label, fn = item if isinstance(item, tuple) else (None, item)
        print(signature_of(fn, label))
