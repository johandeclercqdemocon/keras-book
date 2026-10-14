# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch01c.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
def first(e): return f"{type(e).__name__}: " + [l for l in str(e).splitlines() if l.strip()][1][:80]
x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
def fresh():
    keras.utils.set_random_seed(0); return keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
m = fresh()
print("fresh predict steps=1:", m.predict(x, steps=1, verbose=0).shape, " batch_size=16 steps=1:", m.predict(x, batch_size=16, steps=1, verbose=0).shape)
m = fresh(); print("rank-1 predict:", m.predict(np.array([3.0, 4.0], dtype="float32"), verbose=0).shape)
for kw in ({}, {"steps": 1}):
    try: print("then predict", kw, m.predict(x, verbose=0, **kw).shape)
    except Exception as e: print("then predict", kw, first(e))
m = fresh(); m.predict(x[:3], verbose=0)
try: print("3 rows then 64 rows:", m.predict(x, verbose=0).shape)
except Exception as e: print("3 rows then 64:", first(e))
