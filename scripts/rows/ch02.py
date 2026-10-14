# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch02.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
def tn(t): return ops.convert_to_numpy(t)
print("list int:", ops.convert_to_tensor([1, 2]).dtype, "list float:", ops.convert_to_tensor([1.5]).dtype, "np float64:", ops.convert_to_tensor(np.array([1.5])).dtype)
for v, dt in (([1.7], "int32"), ([300], "uint8")):
    try: print("convert_to_tensor", v, dt, tn(ops.convert_to_tensor(v, dtype=dt)))
    except Exception as e: print("convert_to_tensor", v, dt, first(e))
print("cast 1.7,-1.7 int32:", tn(ops.cast([1.7, -1.7], "int32")))
try: print("cast 'abc':", ops.cast([1.0], "abc"))
except Exception as e: print("cast bad dtype:", first(e))
a = tn(ops.convert_to_tensor(np.array([1.0, 2.0], dtype="float32"))); print("convert_to_numpy writeable:", a.flags.writeable, "owns:", a.flags.owndata)
src = np.array([1.0, 2.0], dtype="float32"); t = ops.convert_to_tensor(src); src[0] = 99; print("tensor after numpy write:", tn(t))
print("normal default dtype:", keras.random.normal((2,)).dtype)
r = tn(keras.random.normal((100000,), mean=5.0, stddev=2.0, seed=1)); print("mean/std:", round(float(r.mean()),3), round(float(r.std()),3))
try: keras.random.normal(3, seed=1); print("shape int: accepted")
except Exception as e: print("normal shape=3:", first(e))
keras.utils.set_random_seed(3); a = tn(keras.random.normal((2,), seed=keras.random.SeedGenerator()))
keras.utils.set_random_seed(3); b = tn(keras.random.normal((2,), seed=keras.random.SeedGenerator()))
print("SeedGenerator(None) after same global seed equal:", bool(np.array_equal(a, b)))
a = tn(keras.random.normal((2,), seed=keras.random.SeedGenerator())); b = tn(keras.random.normal((2,), seed=keras.random.SeedGenerator()))
print("two SeedGenerator(None) equal:", bool(np.array_equal(a, b)))
try: keras.random.SeedGenerator(1, foo=2)
except Exception as e: print("SeedGenerator foo:", first(e))
print("SeedGenerator name:", keras.random.SeedGenerator(1).name, keras.random.SeedGenerator(1, name="noise").name)
for s in (1.5, None, "7", -1, 2**32):
    try: keras.utils.set_random_seed(s); print("set_random_seed", repr(s), "accepted")
    except Exception as e: print("set_random_seed", repr(s), first(e))
