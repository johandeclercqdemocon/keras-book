# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch01.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples  # sets the book env
import io, contextlib
import keras, numpy as np
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1); y = 2 * x + 1
def line(**kw):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1, **kw)]); return m
# Sequential
m = keras.Sequential([keras.layers.Dense(1)]); print("no Input: weights", len(m.weights))
try: m.summary()
except Exception as e: print("summary unbuilt:", first(e))
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)], trainable=False)
m.compile(keras.optimizers.SGD(0.1), "mse"); w0 = m.get_weights()[0].copy(); h = m.fit(x, y, epochs=3, verbose=0)
print("trainable=False changed:", not np.array_equal(w0, m.get_weights()[0]), "loss", [round(v,4) for v in h.history["loss"]])
print("names:", keras.Sequential().name, keras.Sequential().name)
# Input
try:
    m = keras.Sequential([keras.Input((1,), batch_size=16), keras.layers.Dense(1)]); m.compile("sgd","mse")
    m.fit(x, y, batch_size=32, epochs=1, verbose=0); print("batch_size=16 fit 32: accepted")
except Exception as e: print("Input batch_size=16, fit 32:", first(e))
try: keras.Input((1,), batch_shape=(None, 1)); print("shape+batch_shape: accepted")
except Exception as e: print("shape+batch_shape:", first(e))
print("Input name:", keras.Input((1,), name="x").name, keras.Input((1,)).name)
mi = keras.Sequential([keras.Input((1,), dtype="int32"), keras.layers.Dense(1)])
print("int32 input, predict 0.7:", mi.predict(np.array([[0.7]]), verbose=0).ravel(), "kernel", mi.get_weights()[0].ravel(), "bias", mi.get_weights()[1])
# Dense
m = line(use_bias=False); m.compile(keras.optimizers.Adam(0.1),"mse"); h=m.fit(x,y,epochs=100,batch_size=16,verbose=0)
print("use_bias=False: params", m.count_params(), "final loss", round(h.history["loss"][-1],4))
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((1,)), keras.layers.Dense(4, activation="tanh", kernel_initializer="zeros"), keras.layers.Dense(1, kernel_initializer="zeros")])
m.compile(keras.optimizers.Adam(0.05),"mse"); m.fit(x, x**2, epochs=50, verbose=0)
k = m.layers[0].get_weights()[0]; print("zeros init hidden kernel:", np.round(k,4))
m = line(kernel_constraint=keras.constraints.NonNeg()); m.compile(keras.optimizers.Adam(0.1),"mse"); h=m.fit(x,-2*x+1,epochs=100,batch_size=16,verbose=0)
print("NonNeg on y=-2x+1: w", m.get_weights()[0].ravel(), "loss", round(h.history["loss"][-1],4))
m = line(kernel_regularizer=keras.regularizers.L2(0.1)); m.compile(keras.optimizers.Adam(0.1),"mse"); h=m.fit(x,y,epochs=100,batch_size=16,verbose=0)
print("L2(0.1): w", m.get_weights()[0].ravel(), "loss", round(h.history["loss"][-1],4))
keras.utils.set_random_seed(0)
d = keras.layers.Dense(8, lora_rank=2); mm = keras.Sequential([keras.Input((16,)), d])
print("lora_rank=2 on 16->8: trainable", sum(int(np.prod(v.shape)) for v in d.trainable_weights), "non-trainable", sum(int(np.prod(v.shape)) for v in d.non_trainable_weights))
keras.utils.set_random_seed(0); d0 = keras.layers.Dense(8); mm = keras.Sequential([keras.Input((16,)), d0]); print("plain 16->8 trainable", d0.count_params())
try: keras.layers.Dense(1, foo=1)
except Exception as e: print("Dense foo=1:", first(e))
print("Dense name kwarg:", keras.layers.Dense(1, name="out").name)
# compile
m = line()
try: m.compile(optimizer="sgd"); m.fit(x, y, verbose=0); print("no loss: accepted")
except Exception as e: print("no loss:", first(e))
m = line(); m.compile(keras.optimizers.Adam(0.1), "mse", metrics=["mae"]); h=m.fit(x,y,epochs=2,verbose=0); print("metrics keys:", list(h.history))
class Count(keras.callbacks.Callback):
    n = 0
    def on_train_batch_end(self, batch, logs=None): Count.n += 1
for spe in (1, 4):
    Count.n = 0; m = line(); m.compile("sgd", "mse", steps_per_execution=spe); m.fit(x, y, batch_size=16, epochs=1, verbose=0, callbacks=[Count()])
    print(f"steps_per_execution={spe}: batch-end calls {Count.n}, iterations {int(m.optimizer.iterations.numpy())}")
m = line(); m.compile("sgd","mse"); print("jit_compile auto ->", m.jit_compile)
m = line(); m.compile("sgd","mse", run_eagerly=True); h=m.fit(x,y,epochs=1,verbose=0); print("run_eagerly ok, loss", round(h.history["loss"][0],4))
m = line(); m.compile("sgd","mse"); h=m.fit(x,y,epochs=1,verbose=0); print("graph loss", round(h.history["loss"][0],4))
# fit
m = line(); m.compile(keras.optimizers.Adam(0.1),"mse"); h=m.fit(x,y,epochs=3,validation_split=0.25,verbose=0)
print("validation_split last val_loss", round(h.history["val_loss"][-1],6), "evaluate last 16", round(m.evaluate(x[-16:],y[-16:],verbose=0),6), "first 16", round(m.evaluate(x[:16],y[:16],verbose=0),6))
m = line(); m.compile(keras.optimizers.Adam(0.1),"mse"); h=m.fit(x,y,epochs=1,validation_split=0.25,validation_data=(x[:8],y[:8]),verbose=0)
print("both validation:", round(h.history["val_loss"][-1],6), "eval first 8", round(m.evaluate(x[:8],y[:8],verbose=0),6), "eval last 16", round(m.evaluate(x[-16:],y[-16:],verbose=0),6))
m = line(); m.compile("sgd","mse"); h=m.fit(x,y,epochs=5,initial_epoch=3,verbose=0); print("initial_epoch=3 epochs=5 ->", len(h.history["loss"]), "epochs", h.epoch)
m = line(); m.compile("sgd","mse"); m.fit(x,y,epochs=2,batch_size=16,steps_per_epoch=2,verbose=0); print("steps_per_epoch=2, 2 epochs -> iterations", int(m.optimizer.iterations.numpy()))
m = line(); m.compile("sgd","mse"); h=m.fit(x,y,epochs=4,validation_data=(x,y),validation_freq=2,verbose=0); print("validation_freq=2: loss", len(h.history["loss"]), "val", len(h.history["val_loss"]))
w = np.where(x[:,0] > 0, 0.0, 1.0); yy = np.where(x > 0, 10.0, y).astype("float32")
m = line(); m.compile(keras.optimizers.Adam(0.1),"mse"); m.fit(x,yy,sample_weight=w,epochs=100,batch_size=16,verbose=0)
print("sample_weight 0 on x>0 with y=10 there: w,b", m.get_weights()[0].ravel(), m.get_weights()[1])
m = line(); m.compile(keras.optimizers.Adam(0.1),"mse"); m.fit(x,yy,epochs=100,batch_size=16,verbose=0)
print("same data no weights: w,b", m.get_weights()[0].ravel(), m.get_weights()[1])
# predict. The model `m` below first gets a rank-1 array; steps.py shows the same calls on a new model
m = line(); print("predict rank-1 shape:", m.predict(np.array([3.0, 4.0], dtype="float32"), verbose=0).shape)
for kw in ({"steps": 1}, {"batch_size": 10, "steps": 2}):
    try: print("predict", kw, m.predict(x, verbose=0, **kw).shape)
    except Exception as e: print("predict", kw, first(e))
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices(x).batch(16)
try: print("predict dataset steps=2 (same model):", m.predict(ds, steps=2, verbose=0).shape)
except Exception as e: print("predict dataset steps=2 (same model):", first(e))
# evaluate
m = line(); m.compile("sgd","mse", metrics=["mae"]); print("evaluate list:", m.evaluate(x,y,verbose=0)); print("evaluate dict:", m.evaluate(x,y,verbose=0,return_dict=True))
try: m.evaluate(x, y, verbose=0, foo=1)
except Exception as e: print("evaluate foo=1:", first(e))
# summary
lines = []; m = line(); m.summary(print_fn=lambda s, **k: lines.append(s)); print("print_fn calls:", len(lines), "first:", lines[0][:40])
buf = io.StringIO()
with contextlib.redirect_stdout(buf): m.summary(line_length=60, show_trainable=True)
print("show_trainable header:", [l for l in buf.getvalue().splitlines() if "Layer" in l][0])
