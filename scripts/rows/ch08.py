# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch08.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
n = lambda t: ops.convert_to_numpy(t)
keras.utils.set_random_seed(0)
x = np.ones((1, 10000), dtype="float32")
out = n(layers.Dropout(0.5)(x, training=True)); print("dropout 0.5: kept share", round(float((out != 0).mean()),4), "kept values", sorted(set(np.round(out[out != 0], 4).tolist())), "mean", round(float(out.mean()),4))
print("inference unchanged:", bool((n(layers.Dropout(0.5)(x)) == 1).all()))
for r in (1.0, 1.5, -0.1):
    try: layers.Dropout(r); print("rate", r, "accepted")
    except Exception as e: print("rate", r, first(e))
o = n(layers.Dropout(0.5, noise_shape=(4, 1))(np.ones((4, 6), "float32"), training=True)); print("noise_shape (4,1):\n", o)
a, b = layers.Dropout(0.5, seed=1), layers.Dropout(0.5, seed=1)
ma, mb = n(a(x, training=True)), n(b(x, training=True)); ma2 = n(a(x, training=True))
print("same seed two layers equal masks:", bool(np.array_equal(ma, mb)), " one layer second call equal:", bool(np.array_equal(ma, ma2)))
d = layers.Dense(2, kernel_initializer="ones", kernel_regularizer=keras.regularizers.L2()); d(np.ones((1, 5), "float32")); print("L2 default penalty on 10 ones:", [round(float(v),4) for v in d.losses])
print("L2 default value:", keras.regularizers.L2().l2)
for v in (-0.1, "0.1"):
    try: r = keras.regularizers.L2(v); print("L2", repr(v), "accepted", r.l2)
    except Exception as e: print("L2", repr(v), first(e))
g = layers.GaussianNoise(0.3); z = np.zeros((1, 100000), "float32")
print("noise std in training:", round(float(n(g(z, training=True)).std()),4), " inference max abs:", float(np.abs(n(g(z))).max()))
try: layers.GaussianNoise(-1.0); print("negative stddev accepted")
except Exception as e: print("negative stddev:", first(e))
g1, g2 = layers.GaussianNoise(0.3, seed=4), layers.GaussianNoise(0.3, seed=4)
print("same seed equal noise:", bool(np.array_equal(n(g1(z, training=True)), n(g2(z, training=True)))))
