# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch03.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
keras.utils.set_random_seed(0)
class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs); self.units = units
    def build(self, input_shape):
        self.seen = input_shape
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")
    def call(self, x): return ops.matmul(x, self.kernel) + self.bias
print("name:", MyDense(3).name)
try: keras.layers.Layer(foo=1)
except Exception as e: print("Layer foo:", first(e))
class Ident(keras.layers.Layer):
    def call(self, x): return x
l = Ident(activity_regularizer=keras.regularizers.L2(0.01)); l(np.ones((1, 4), dtype="float32")); print("activity L2 on ones(1,4):", [round(float(v), 4) for v in l.losses])
l = MyDense(2, dtype="float64"); out = l(np.ones((1, 3), dtype="float32")); print("dtype float64: kernel", l.kernel.dtype, "out", out.dtype)
l = MyDense(2); out = l(np.ones((1, 3), dtype="float64")); print("autocast default, float64 input -> out", out.dtype)
l = MyDense(2, autocast=False)
try: out = l(np.ones((1, 3), dtype="float64")); print("autocast=False float64 input -> out", out.dtype)
except Exception as e: print("autocast=False float64 input:", first(e))
# add_weight
class W(keras.layers.Layer):
    def __init__(self, mode, **kw): super().__init__(**kw); self.mode = mode
    def build(self, s):
        if self.mode == "positional": self.w = self.add_weight((2,), "ones")
        elif self.mode == "noshape": self.w = self.add_weight(name="w")
        elif self.mode == "noinit": self.w = self.add_weight(shape=(200, 300))
        elif self.mode == "noname": self.w = self.add_weight(shape=(2,))
        elif self.mode == "int": self.w = self.add_weight(shape=(3,), dtype="int32")
    def call(self, x): return x
for mode in ("positional", "noshape", "noinit", "noname", "int"):
    try:
        l = W(mode); l(np.ones((1, 2), dtype="float32")); v = ops.convert_to_numpy(l.w)
        print(f"add_weight {mode}: shape {tuple(l.w.shape)} dtype {l.w.dtype} path {l.w.path} min {v.min():.4f} max {v.max():.4f}")
    except Exception as e: print("add_weight", mode, first(e))
print("glorot limit for (200,300):", round(float(np.sqrt(6 / 500)), 4))
class C(keras.layers.Layer):
    def build(self, s): self.w = self.add_weight(shape=(3,), initializer="ones", constraint=keras.constraints.NonNeg())
    def call(self, x): return x * self.w
m = keras.Sequential([keras.Input((3,)), C(), layers.Dense(1)]); m.compile(keras.optimizers.SGD(1.0), "mse")
m.fit(np.ones((8,3),"float32"), -10*np.ones((8,1),"float32"), epochs=5, verbose=0); print("constraint NonNeg weight after fit:", ops.convert_to_numpy(m.layers[0].w))
# build input_shape
l = MyDense(3); l(np.ones((2, 4), dtype="float32")); print("build saw (direct call):", l.seen)
l = MyDense(3); keras.Sequential([keras.Input((4,)), l]); print("build saw (in model):", l.seen)
# BatchNormalization
x = np.random.default_rng(0).normal(5.0, 1.0, size=(64, 4)).astype("float32")
bn = layers.BatchNormalization(); bn(x)
print("BN params total/trainable:", bn.count_params(), len(bn.trainable_weights), len(bn.non_trainable_weights))
for steps in (1, 100, 460):
    bn = layers.BatchNormalization(); bn(x)
    for _ in range(steps): bn(x, training=True)
    mm = float(ops.convert_to_numpy(bn.moving_mean)[0]); print(f"momentum 0.99: moving_mean after {steps} steps: {mm:.3f} (data mean {x[:,0].mean():.3f})")
bn = layers.BatchNormalization()
for _ in range(10): bn(x, training=True)
inf = ops.convert_to_numpy(bn(x, training=False)); tr = ops.convert_to_numpy(bn(x, training=True))
print("after 10 steps: inference output mean", round(float(inf.mean()),3), "training output mean", round(float(tr.mean()),3))
bn = layers.BatchNormalization(momentum=0.9)
for _ in range(10): bn(x, training=True)
print("momentum 0.9, 10 steps moving_mean:", round(float(ops.convert_to_numpy(bn.moving_mean)[0]),3))
bn = layers.BatchNormalization(axis=0); bn(x)
print("axis=0 weights shape:", tuple(bn.gamma.shape))
try: bn(x[:32]); print("axis=0 batch 32: accepted")
except Exception as e: print("axis=0 then batch 32:", first(e))
for kw in ({"center": False}, {"scale": False}, {"renorm": True}):
    b = layers.BatchNormalization(**kw); b(x); print(kw, "trainable", [w.name for w in b.trainable_weights], "non-trainable", [w.name for w in b.non_trainable_weights])
c = np.full((64, 1), 3.0, dtype="float32"); b = layers.BatchNormalization(); print("constant feature, training output:", float(ops.convert_to_numpy(b(c, training=True)).max()))
try: layers.BatchNormalization(foo=1)
except Exception as e: print("BN foo:", first(e))
