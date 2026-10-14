# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch15.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
x = np.ones((1, 4), "float32")
for pol in ("mixed_float16", "float16", "mixed_bfloat16", "float32"):
    keras.mixed_precision.set_global_policy(pol); d = layers.Dense(2); o = d(x)
    print(f"{pol:15s} kernel {d.kernel.dtype:8s} output {o.dtype.name if hasattr(o.dtype,'name') else o.dtype}")
keras.mixed_precision.set_global_policy("mixed_float16")
d_after = layers.Dense(2); print("policy stays for later layers:", d_after.dtype_policy.name, "global:", keras.mixed_precision.global_policy().name)
keras.mixed_precision.set_global_policy("float32")
for bad in ("fp16", None):
    try: keras.mixed_precision.set_global_policy(bad); print(repr(bad), "accepted ->", keras.mixed_precision.global_policy().name)
    except Exception as e: print(repr(bad), first(e))
keras.mixed_precision.set_global_policy("float32")
