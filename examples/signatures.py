"""Print the signature of Keras callables, as the installed version defines them.

Each chapter prints a signature in a `#### name` block. `signature_text` makes that text, and the checker
compares the block with it, so a signature in the book is produced by the library, not typed by a person.

A name is dotted from `keras`, for example `keras.layers.Dense` or `keras.Model.fit`. For a class, the
signature is the signature of its constructor. Keras defines `Model.fit` on a back-end `Trainer` class, so
the book names a method by the class that the reader calls it on, not by the class that defines it.
"""
import importlib
import inspect
import textwrap


def resolve(path: str):
    """Turn a dotted name such as `keras.layers.Dense` or `keras.Model.fit` into the object.

    The longest prefix that imports as a module is the module. The rest are attributes.
    """
    parts = path.split(".")
    if parts[0] != "keras":
        raise ValueError(f"{path}: the first name must be keras")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for attr in parts[k:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(path)


def _signature(obj) -> inspect.Signature:
    """The signature that a caller sees: the constructor for a class, and no `self`."""
    sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return sig.replace(parameters=params)


def _plain(p: inspect.Parameter) -> str:
    """One parameter as the caller writes it: no annotation, and the default in its repr."""
    stars = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "")
    return stars + p.name + ("" if p.default is p.empty else f"={p.default!r}")


def signature_text(obj, name: str) -> str:
    """`name(` then one parameter for each line, with the `/` and `*` markers, then `)`.

    The annotations are left out: the parameter table of the chapter says what each parameter takes.
    """
    params = list(_signature(obj).parameters.values())
    kinds = [p.kind for p in params]
    lines = []
    for i, p in enumerate(params):
        if p.kind is p.KEYWORD_ONLY and (i == 0 or kinds[i - 1] not in (p.KEYWORD_ONLY, p.VAR_POSITIONAL)):
            lines.append("*")
        lines.append(_plain(p))
        if p.kind is p.POSITIONAL_ONLY and (i + 1 == len(kinds) or kinds[i + 1] is not p.POSITIONAL_ONLY):
            lines.append("/")
    if not lines:
        return f"{name}()"
    return f"{name}(\n" + "".join(f"    {line},\n" for line in lines) + ")"


def parameters(obj) -> list:
    """(name, default) for each parameter that a caller can pass. The default is its repr, None if required.

    A `*args` or `**kwargs` parameter keeps its stars and has the default "—".
    """
    out = []
    for p in _signature(obj).parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            out.append(("*" + p.name, "—"))
        elif p.kind is p.VAR_KEYWORD:
            out.append(("**" + p.name, "—"))
        else:
            out.append((p.name, None if p.default is p.empty else repr(p.default)))
    return out


def signature_of(fn, label=None) -> str:
    """`label(parameters)` on one line, wrapped at 78 characters. Used by the exercises."""
    text = ", ".join(str(p) for p in _signature(fn).parameters.values())
    name = label or getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    return textwrap.fill(f"{name}({text})", width=78, subsequent_indent="    ",
                         break_long_words=False, break_on_hyphens=False)


def show(*items) -> None:
    for item in items:
        label, fn = item if isinstance(item, tuple) else (None, item)
        print(signature_of(fn, label))
