# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch14.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, os, sys, tempfile, warnings, keras, numpy as np
from keras import layers
warnings.simplefilter("ignore")
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
def tryp(label, fn):
    try: print(label, fn())
    except Exception as e: print(label, first(e))
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((4,)), layers.Dense(2)]); m.compile("adam", "mse"); m.fit(np.zeros((4, 4)), np.zeros((4, 2)), verbose=0)
d = tempfile.mkdtemp(); P = lambda f: os.path.join(d, f)
tryp("save no suffix:", lambda: m.save(P("model")))
tryp("save .h5:", lambda: (m.save(P("model.h5")), os.path.exists(P("model.h5")))[1])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); r = None
    try: m.save(P("m2.keras"), save_format="tf"); r = "accepted"
    except Exception as e: r = first(e)
    print("save save_format=tf:", r, "warnings:", [str(x.message)[:100] for x in w], "file:", os.path.exists(P("m2.keras")))
m.save(P("a.keras")); t0 = os.path.getmtime(P("a.keras")); m.save(P("a.keras")); print("overwrite default: no question, file rewritten")
sys.stdin = open(os.devnull)
tryp("overwrite=False on an existing file, no terminal:", lambda: m.save(P("a.keras"), overwrite=False))
tryp("zipped=False:", lambda: (m.save(P("unzipped.keras"), zipped=False)))
tryp("zipped=False no suffix:", lambda: (m.save(P("unzipped"), zipped=False), os.path.isdir(P("unzipped")), sorted(os.listdir(P("unzipped"))) if os.path.isdir(P("unzipped")) else None)[1:])
l = keras.saving.load_model(P("a.keras"), compile=False); print("compile=False: compiled", l.compiled, "has optimizer attr", hasattr(l, "optimizer"))
tryp("fit after compile=False:", lambda: l.fit(np.zeros((4, 4)), np.zeros((4, 2)), verbose=0))
lm = keras.Sequential([keras.Input((4,)), layers.Lambda(lambda x: x * 2.0), layers.Dense(1)]); lm.save(P("lambda.keras"))
tryp("load Lambda safe_mode=True:", lambda: keras.saving.load_model(P("lambda.keras")))
tryp("load Lambda safe_mode=False:", lambda: type(keras.saving.load_model(P("lambda.keras"), safe_mode=False).layers[0]).__name__)
tryp("save_weights .h5:", lambda: m.save_weights(P("w.h5")))
m.save_weights(P("w.weights.h5")); print("save_weights ok:", os.path.exists(P("w.weights.h5")))
big = keras.Sequential([keras.Input((100,)), layers.Dense(100), layers.Dense(100)])
tryp("max_shard_size tiny:", lambda: (big.save_weights(P("s.weights.h5"), max_shard_size=0.00005), sorted(f for f in os.listdir(d) if f.startswith("s")))[1])
other = keras.Sequential([keras.Input((4,)), layers.Dense(3)])
tryp("load_weights mismatch:", lambda: other.load_weights(P("w.weights.h5")))
before = other.get_weights()[0].copy()
tryp("load_weights skip_mismatch:", lambda: (other.load_weights(P("w.weights.h5"), skip_mismatch=True), bool(np.array_equal(before, other.get_weights()[0])))[1])
tryp("load_weights .keras file:", lambda: (keras.Sequential([keras.Input((4,)), layers.Dense(2)]).load_weights(P("a.keras")), "ok")[1])
tryp("export onnx:", lambda: m.export(P("m.onnx"), format="onnx"))
tryp("export bad format:", lambda: m.export(P("m.x"), format="xyz"))
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf): m.export(P("saved"), verbose=None)
print("export verbose=None printed lines:", len(buf.getvalue().splitlines()))
buf = io.StringIO()
with contextlib.redirect_stdout(buf): m.export(P("saved2"), verbose=False)
print("export verbose=False printed lines:", len(buf.getvalue().splitlines()))
@keras.saving.register_keras_serializable()
class Twice(layers.Layer):
    def call(self, x): return 2 * x
@keras.saving.register_keras_serializable(package="book", name="double")
class Twice2(layers.Layer):
    def call(self, x): return 2 * x
print("registered names:", keras.saving.get_registered_name(Twice), keras.saving.get_registered_name(Twice2))
def redefine():
    @keras.saving.register_keras_serializable()
    class Twice(layers.Layer):
        def call(self, x): return 3 * x
    return Twice
tryp("register same name twice:", lambda: keras.saving.get_registered_object("Custom>Twice") is redefine())
