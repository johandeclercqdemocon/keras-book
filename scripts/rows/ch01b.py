# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch01b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, io, contextlib, keras, numpy as np
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1); y = 2 * x + 1
def line():
    keras.utils.set_random_seed(0); return keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
m = line(); m.compile("sgd","mse", metrics=["mae"]); print("evaluate list:", m.evaluate(x,y,verbose=0)); print("evaluate dict:", m.evaluate(x,y,verbose=0,return_dict=True))
try: m.evaluate(x, y, verbose=0, foo=1); print("evaluate foo=1 accepted")
except Exception as e: print("evaluate foo=1:", first(e))
print("evaluate steps=1 bs=16:", m.evaluate(x, y, batch_size=16, steps=1, verbose=0), "first 16 only:", m.evaluate(x[:16], y[:16], verbose=0))
lines = []; m.summary(print_fn=lambda s, **k: lines.append(s)); print("print_fn calls:", len(lines), repr(lines[0][:60]))
buf = io.StringIO()
with contextlib.redirect_stdout(buf): m.summary(line_length=60, show_trainable=True)
print(buf.getvalue())
buf = io.StringIO()
with contextlib.redirect_stdout(buf): m.summary()
print("default width:", max(len(l) for l in buf.getvalue().splitlines()))
