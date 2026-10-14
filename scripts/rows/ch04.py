# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch04.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
i = keras.Input((4,)); o = layers.Dense(2)(i)
print("positional:", keras.Model(i, o).count_params(), "keywords:", keras.Model(inputs=i, outputs=o, name="m").name)
for kw in ({"inputs": i}, {"inputs": i, "outputs": o, "foo": 1}):
    try: keras.Model(**kw); print(kw.keys(), "accepted")
    except Exception as e: print(list(kw), first(e))
try: keras.Model(i, o, "named"); print("third positional accepted")
except Exception as e: print("third positional:", first(e))
print("Model() empty:", type(keras.Model()).__name__, keras.Model().built)
a, b = keras.Input((8,)), keras.Input((8,))
print("concat -1:", layers.Concatenate()([a, b]).shape)
c = keras.Input((3,))
try: print("concat (8),(3):", layers.Concatenate()([a, c]).shape)
except Exception as e: print("concat 8,3:", first(e))
try: print("concat axis=0 (8),(3):", layers.Concatenate(axis=0)([a, c]).shape)
except Exception as e: print("concat axis 0:", first(e))
d, e2 = keras.Input((2, 5)), keras.Input((3, 5))
try: print("concat axis=1 (2,5),(3,5):", layers.Concatenate(axis=1)([d, e2]).shape)
except Exception as e: print("axis1:", first(e))
try: print("concat axis=-1 (2,5),(3,5):", layers.Concatenate()([d, e2]).shape)
except Exception as e: print("axis -1 mismatch:", first(e))
try: layers.Concatenate()([a]); print("one input accepted")
except Exception as e: print("one input:", first(e))
