# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch06.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
from examples.digits import load
x_train, x_test, y_train, y_test = load()
f = lambda t: float(ops.convert_to_numpy(t))
rng = np.random.default_rng(0); scores = rng.normal(size=(8, 10)).astype("float32"); labels = np.array([0, 3, 0, 5, 1, 0, 9, 2])
L = keras.losses.SparseCategoricalCrossentropy
per = ops.convert_to_numpy(L(from_logits=True, reduction=None)(labels, scores))
print("reduction None shape:", per.shape, "mean", round(float(per.mean()),4), "sum", round(float(per.sum()),4))
print("default:", round(f(L(from_logits=True)(labels, scores)),4), "sum:", round(f(L(from_logits=True, reduction="sum")(labels, scores)),4))
ig = f(L(from_logits=True, ignore_class=0)(labels, scores)); keep = labels != 0
print("ignore_class=0:", round(ig,4), "mean over the 5 others:", round(float(per[keep].mean()),4), "sum/8:", round(float(per[keep].sum()/8),4))
print("name:", L().name)
def run(opt, epochs=10, metrics=("accuracy",)):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    m.compile(opt, L(from_logits=True), metrics=list(metrics))
    h = m.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test), verbose=0)
    return m, h
for name, opt in [("SGD 0.1 nesterov m0.9", keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True)),
                  ("SGD 0.1 momentum 0.9", keras.optimizers.SGD(0.1, momentum=0.9))]:
    m, h = run(opt); print(name, "epoch1", round(h.history["val_accuracy"][0],3), "epoch10", round(h.history["val_accuracy"][-1],3))
for wd in (None, 0.1):
    m, h = run(keras.optimizers.SGD(0.1, weight_decay=wd)); print("SGD weight_decay", wd, "kernel norm", round(float(np.linalg.norm(m.layers[0].get_weights()[0])),3), "val acc", round(h.history["val_accuracy"][-1],3))
m, h = run(keras.optimizers.SGD(0.1, use_ema=True)); print("use_ema val acc", round(h.history["val_accuracy"][-1],3), "evaluate after fit", round(m.evaluate(x_test, y_test, verbose=0)[1],3))
m, h = run(keras.optimizers.SGD(0.1, gradient_accumulation_steps=4), epochs=1)
print("gradient_accumulation_steps=4, 1 epoch of 43 batches: iterations", int(m.optimizer.iterations.numpy()))
m, h = run(keras.optimizers.SGD(0.1), epochs=1); print("plain: iterations", int(m.optimizer.iterations.numpy()))
for kw in ({"lr": 0.1}, {"foo": 1}):
    try: keras.optimizers.SGD(**kw); print(kw, "accepted")
    except Exception as e: print("SGD", kw, first(e))
for name, opt in [("Adam default", keras.optimizers.Adam()), ("Adam beta_1=0", keras.optimizers.Adam(beta_1=0.0)),
                  ("Adam epsilon=1.0", keras.optimizers.Adam(epsilon=1.0)), ("Adam amsgrad", keras.optimizers.Adam(amsgrad=True))]:
    m, h = run(opt); print(name, "epoch1", round(h.history["val_accuracy"][0],3), "epoch10", round(h.history["val_accuracy"][-1],3))
CD = keras.optimizers.schedules.CosineDecay
s = CD(0.01, decay_steps=100); print("cosine 0,100,200:", [round(float(s(i)),5) for i in (0, 100, 200)])
s = CD(0.01, decay_steps=100, alpha=0.1); print("alpha 0.1 at 100, 200:", [round(float(s(i)),5) for i in (100, 200)])
s = CD(0.001, decay_steps=100, warmup_target=0.01, warmup_steps=20); print("warmup 0,10,20,120:", [round(float(s(i)),5) for i in (0, 10, 20, 70, 120)])
try: s = CD(0.001, decay_steps=100, warmup_steps=20); print("warmup_steps without target:", [round(float(s(i)),5) for i in (0, 10, 20, 120)])
except Exception as e: print("warmup without target:", first(e))
m, h = run("adam", metrics=("accuracy", keras.metrics.SparseTopKCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top1")))
print({k: round(v[-1],4) for k, v in h.history.items() if k.startswith("val_")})
print("to_categorical [-1, 1] num_classes 3:", keras.utils.to_categorical([-1, 1], num_classes=3).tolist())
