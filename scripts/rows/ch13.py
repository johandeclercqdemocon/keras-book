# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch13.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
keras.utils.set_random_seed(0)
base = keras.Sequential([keras.Input((8, 8, 1)), layers.Conv2D(4, 3, padding="same"), layers.GlobalMaxPooling2D()], name="base")
outer = keras.Sequential([base, layers.Dense(3)])
c = keras.models.clone_model(base)
print("clone weights equal original:", all(np.array_equal(a, b) for a, b in zip(c.get_weights(), base.get_weights())), " same layer objects:", c.layers[0] is base.layers[0])
s = keras.models.clone_model(base, clone_function=lambda layer: layer)
print("clone_function=identity: same layer objects:", s.layers[0] is base.layers[0])
oc = keras.models.clone_model(outer); print("nested default: inner model new object:", oc.layers[0] is not base, " inner conv new:", oc.layers[0].layers[0] is not base.layers[0])
names = []
def fn(layer): names.append(layer.name); return layer.__class__.from_config(layer.get_config())
for rec in (False, True):
    names.clear(); keras.models.clone_model(outer, clone_function=fn, recursive=rec); print("recursive", rec, "clone_function saw:", names)
try: keras.models.clone_model(base, foo=1); print("foo accepted")
except Exception as e: print("clone_model foo:", first(e))
w = base.get_weights(); print("get_weights: types", sorted({type(a).__name__ for a in w}), "count", len(w), "shapes", [a.shape for a in w])
w[0][...] = 0; print("editing the returned array changed the model:", bool((base.get_weights()[0] == 0).all()))
for label, val in [("too short", base.get_weights()[:1]), ("wrong shape", [np.zeros((3, 3, 1, 5), "float32"), np.zeros(5, "float32")]), ("too long", base.get_weights() + [np.zeros(1)])]:
    try: base.set_weights(val); print("set_weights", label, "accepted")
    except Exception as e: print("set_weights", label, first(e))
