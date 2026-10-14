# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch09.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:130]}"
f = lambda t: float(ops.convert_to_numpy(t))
class Sq(keras.losses.Loss):
    def call(self, y_true, y_pred): return ops.square(y_true - y_pred)
yt = np.zeros((4, 1), "float32"); yp = np.array([[1.], [2.], [3.], [4.]], "float32")
print("Loss name default:", Sq().name, "| mean:", f(Sq()(yt, yp)), "sum:", f(Sq(reduction="sum")(yt, yp)))
try: print("reduction None shape:", ops.convert_to_numpy(Sq(reduction=None)(yt, yp)).shape)
except Exception as e: print("reduction None:", first(e))
try: Sq(reduction="mean"); print("reduction 'mean' accepted:", f(Sq(reduction="mean")(yt, yp)))
except Exception as e: print("reduction mean:", first(e))
class M(keras.metrics.Metric):
    def __init__(self, **kw):
        super().__init__(**kw); self.total = self.add_variable(shape=(), initializer="zeros", name="total")
    def update_state(self, y_true, y_pred, sample_weight=None): self.total.assign(self.total + ops.cast(ops.shape(y_true)[0], "float32"))
    def result(self): return self.total
print("Metric name default:", M().name)
class PyState(keras.metrics.Metric):
    def __init__(self, **kw): super().__init__(name="py_count", **kw); self.n = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None): self.n += 1.0
    def result(self): return ops.convert_to_tensor(self.n)
    def reset_state(self): self.n = 0.0
from examples.digits import load
x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((64,)), layers.Dense(10)])
m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[PyState(), M(name="seen")])
h = m.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
print("python-state metric per epoch:", h.history["py_count"], " variable metric:", h.history["seen"], " batches per epoch 43")
try:
    class NoInit(keras.metrics.Metric):
        def __init__(self, **kw): super().__init__(**kw); self.v = self.add_variable(shape=(2,), name="v")
    NoInit(); print("add_variable without initializer accepted")
except Exception as e: print("add_variable no initializer:", first(e))
v = M().total; print("add_variable aggregation default:", v.aggregation, "dtype", v.dtype)
keras.utils.set_random_seed(0)
r = keras.Sequential([keras.Input((64,)), layers.Dense(10, kernel_regularizer=keras.regularizers.L2(0.1))])
r.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
xb, yb = x_test[:32], y_test[:32]; yp = r(xb)
lt, lf = f(r.compute_loss(x=xb, y=yb, y_pred=yp, training=True)), f(r.compute_loss(x=xb, y=yb, y_pred=yp, training=False))
print("compute_loss training True/False:", round(lt,4), round(lf,4), "penalty", round(f(sum(r.losses)),4))
try: print("compute_loss without y_pred:", r.compute_loss(x=xb, y=yb))
except Exception as e: print("compute_loss no y_pred:", first(e))
w = np.zeros(32, "float32"); w[:16] = 1
print("compute_loss sample_weight half zero:", round(f(r.compute_loss(x=xb, y=yb, y_pred=yp, sample_weight=w)),4), "first 16 only:", round(f(r.compute_loss(x=xb[:16], y=yb[:16], y_pred=yp[:16])),4))
r.fit(x_train[:64], y_train[:64], epochs=1, verbose=0)
d1 = {k: round(float(v), 4) for k, v in r.compute_metrics(xb, yb, yp).items()}
d2 = {k: round(float(v), 4) for k, v in r.compute_metrics(xb, yb, yp).items()}
print("compute_metrics:", d1, "second call:", d2)
for mm in r.metrics: mm.reset_state()
print("after reset:", {k: round(float(v), 4) for k, v in r.compute_metrics(xb, yb, yp).items()})
w1 = keras.Variable(np.array([1.0], "float32")); w2 = keras.Variable(np.array([1.0], "float32"))
opt = keras.optimizers.SGD(0.1)
try: opt.apply([ops.convert_to_tensor([1.0])]); print("apply without variables before build: ok")
except Exception as e: print("apply no variables, unbuilt:", first(e))
opt = keras.optimizers.SGD(0.1)
try: opt.apply([ops.convert_to_tensor([1.0])], [w1, w2]); print("apply 1 grad 2 vars: ok", ops.convert_to_numpy(w1), ops.convert_to_numpy(w2))
except Exception as e: print("apply 1 grad, 2 vars:", first(e))
opt = keras.optimizers.SGD(0.1); opt.apply([ops.convert_to_tensor([1.0])], [w1])
opt.apply([ops.convert_to_tensor([1.0])]); print("second apply without variables after build: w1 =", ops.convert_to_numpy(w1), "iterations", int(opt.iterations.numpy()))
