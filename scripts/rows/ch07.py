# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch07.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, os, tempfile, warnings, keras, numpy as np
from keras import layers
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:130]}"
from examples.digits import load
x_train, x_test, y_train, y_test = load()
rows = np.random.default_rng(0).choice(len(x_train), 200, replace=False)
xs, ys = x_train[rows], y_train[rows]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def model(metrics=("accuracy",)):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input((64,)), layers.Dense(256, activation="relu"), layers.Dense(256, activation="relu"), layers.Dense(10)])
    m.compile("adam", loss, metrics=list(metrics)); return m
def run(cb, epochs=60, metrics=("accuracy",)):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = model(metrics).fit(xs, ys, epochs=epochs, batch_size=32, validation_data=(x_test, y_test), verbose=0, callbacks=[cb])
    return h, [str(x.message)[:100] for x in w]
ES = keras.callbacks.EarlyStopping
for label, cb in [("patience=0 default", ES()), ("patience=5", ES(patience=5)), ("patience=5 min_delta=0.01", ES(patience=5, min_delta=0.01)),
                  ("typo val_los", ES(monitor="val_los", patience=5)), ("baseline 0.1 patience 5", ES(patience=5, baseline=0.1)),
                  ("start_from_epoch=20 patience 0", ES(start_from_epoch=20))]:
    try: h, w = run(cb); print(f"ES {label}: epochs run {len(h.history['loss'])}", ("warnings: " + str(w)) if w else "")
    except Exception as e: print("ES", label, first(e))
try:
    h, w = run(ES(monitor="val_top1", patience=5), metrics=("accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top1")))
    v = h.history["val_top1"]; print("ES monitor val_top1 auto: epochs run", len(v), "best epoch by max", int(np.argmax(v))+1, w)
except Exception as e: print("ES val_top1 auto:", first(e))
h, w = run(ES(monitor="val_accuracy", patience=5)); v = h.history["val_accuracy"]; print("ES val_accuracy auto: epochs", len(v), "argmax", int(np.argmax(v))+1)
MC = keras.callbacks.ModelCheckpoint; d = tempfile.mkdtemp()
for fp, kw in [("m.h5", {}), ("m.keras", {"save_weights_only": True}), ("w.weights.h5", {"save_weights_only": True})]:
    try: run(MC(os.path.join(d, fp), **kw), epochs=2); print("MC", fp, kw, "ok, exists", os.path.exists(os.path.join(d, fp)))
    except Exception as e: print("MC", fp, kw, first(e))
run(MC(os.path.join(d, "e{epoch:02d}.keras")), epochs=3); print("MC formatted files:", sorted(f for f in os.listdir(d) if f.startswith("e")))
run(MC(os.path.join(d, "t.keras"), save_best_only=True, initial_value_threshold=0.1), epochs=10); print("MC threshold 0.1 file exists:", os.path.exists(os.path.join(d, "t.keras")))
run(MC(os.path.join(d, "every.keras"), save_freq=7), epochs=2); print("MC save_freq=7 exists:", os.path.exists(os.path.join(d, "every.keras")))
RL = keras.callbacks.ReduceLROnPlateau
for kw in ({"epsilon": 1e-4}, {"foo": 1}):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always"); RL(**kw)
        print("RLROP", kw, "accepted", [str(x.message)[:90] for x in w])
    except Exception as e: print("RLROP", kw, first(e))
print("RLROP defaults factor/patience:", RL().factor, RL().patience)
h, _ = run(RL(factor=0.5, patience=1), epochs=15); print("RLROP patience1 factor .5 lr:", [round(v,5) for v in h.history["learning_rate"]])
h, _ = run(RL(factor=0.5, patience=1, cooldown=3), epochs=15); print("RLROP + cooldown 3 lr:", [round(v,5) for v in h.history["learning_rate"]])
x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
keras.utils.set_random_seed(0); m = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)]); m.compile(keras.optimizers.SGD(1.5), "mse")
try: m.fit(x, 2*x+1, epochs=100, batch_size=16, verbose=0, callbacks=[keras.callbacks.TerminateOnNaN(raise_error=True)]); print("raise_error=True: no exception")
except Exception as e: print("TerminateOnNaN raise_error=True:", first(e))
